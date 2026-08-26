"""Tests for feishu_comment — event filtering, access control integration, wiki reverse lookup."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from plugins.platforms.feishu.feishu_comment import (
    parse_drive_comment_event,
    _ALLOWED_NOTICE_TYPES,
    _run_comment_agent,
    _run_comment_agent_with_context,
    _sanitize_comment_text,
)


def _make_event(
    comment_id="c1",
    reply_id="r1",
    notice_type="add_reply",
    file_token="docx_token",
    file_type="docx",
    from_open_id="ou_user",
    to_open_id="ou_bot",
    is_mentioned=True,
):
    """Build a minimal drive comment event SimpleNamespace."""
    return SimpleNamespace(event={
        "event_id": "evt_1",
        "comment_id": comment_id,
        "reply_id": reply_id,
        "is_mentioned": is_mentioned,
        "timestamp": "1713200000",
        "notice_meta": {
            "file_token": file_token,
            "file_type": file_type,
            "notice_type": notice_type,
            "from_user_id": {"open_id": from_open_id},
            "to_user_id": {"open_id": to_open_id},
        },
    })


class TestParseEvent(unittest.TestCase):
    def test_parse_valid_event(self):
        evt = _make_event()
        parsed = parse_drive_comment_event(evt)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["comment_id"], "c1")
        self.assertEqual(parsed["file_type"], "docx")
        self.assertEqual(parsed["from_open_id"], "ou_user")
        self.assertEqual(parsed["to_open_id"], "ou_bot")


class TestEventFiltering(unittest.TestCase):
    """Test the filtering logic in handle_drive_comment_event."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule")
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed")
    def test_self_reply_filtered(self, mock_allowed, mock_resolve, mock_load):
        """Events where from_open_id == self_open_id should be dropped."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event

        evt = _make_event(from_open_id="ou_bot", to_open_id="ou_bot")
        self._run(handle_drive_comment_event(Mock(), evt, self_open_id="ou_bot"))
        mock_load.assert_not_called()

    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule")
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed")
    def test_wrong_receiver_filtered(self, mock_allowed, mock_resolve, mock_load):
        """Events where to_open_id != self_open_id should be dropped."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event

        evt = _make_event(to_open_id="ou_other_bot")
        self._run(handle_drive_comment_event(Mock(), evt, self_open_id="ou_bot"))
        mock_load.assert_not_called()


class TestAccessControlIntegration(unittest.TestCase):
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @patch("plugins.platforms.feishu.feishu_comment_rules.has_wiki_keys", return_value=False)
    @patch("plugins.platforms.feishu.feishu_comment_rules.is_user_allowed", return_value=False)
    @patch("plugins.platforms.feishu.feishu_comment_rules.resolve_rule")
    @patch("plugins.platforms.feishu.feishu_comment_rules.load_config")
    def test_denied_user_no_side_effects(self, mock_load, mock_resolve, mock_allowed, mock_wiki_keys):
        """Denied user should not trigger typing reaction or agent."""
        from plugins.platforms.feishu.feishu_comment import handle_drive_comment_event
        from plugins.platforms.feishu.feishu_comment_rules import ResolvedCommentRule

        mock_resolve.return_value = ResolvedCommentRule(True, "allowlist", frozenset(), "top")
        mock_load.return_value = Mock()

        client = Mock()
        evt = _make_event()
        self._run(handle_drive_comment_event(client, evt, self_open_id="ou_bot"))

        # No API calls should be made for denied users
        client.request.assert_not_called()


class TestSanitizeCommentText(unittest.TestCase):
    def test_angle_brackets_escaped(self):
        self.assertEqual(_sanitize_comment_text("List<String>"), "List&lt;String&gt;")

    def test_ampersand_escaped_first(self):
        self.assertEqual(_sanitize_comment_text("a & b"), "a &amp; b")

    def test_ampersand_not_double_escaped(self):
        result = _sanitize_comment_text("a < b & c > d")
        self.assertEqual(result, "a &lt; b &amp; c &gt; d")
        self.assertNotIn("&amp;lt;", result)
        self.assertNotIn("&amp;gt;", result)


class TestCommentAgentRuntime(unittest.TestCase):
    @patch(
        "plugins.platforms.feishu.feishu_comment._resolve_comment_agent_toolsets",
        return_value=(
            ["feishu_doc", "feishu_drive", "file", "skills", "terminal"],
            ["memory"],
        ),
    )
    @patch(
        "plugins.platforms.feishu.feishu_comment._resolve_model_and_runtime",
        return_value=("test-model", {"provider": "test-provider"}),
    )
    @patch("run_agent.AIAgent")
    def test_uses_feishu_platform_tools_and_project_context(
        self,
        mock_agent_cls,
        _mock_runtime,
        _mock_toolsets,
    ):
        agent = Mock()
        agent.run_conversation.return_value = {
            "final_response": "done",
            "api_calls": 0,
            "messages": [],
        }
        mock_agent_cls.return_value = agent

        result = _run_comment_agent(
            "prompt",
            Mock(),
            session_key="comment-doc:docx:token",
            user_id="ou_user",
        )

        self.assertEqual(result, "done")
        kwargs = mock_agent_cls.call_args.kwargs
        self.assertEqual(
            kwargs["enabled_toolsets"],
            ["feishu_doc", "feishu_drive", "file", "skills", "terminal"],
        )
        self.assertEqual(kwargs["disabled_toolsets"], ["memory"])
        self.assertFalse(kwargs["skip_context_files"])
        self.assertTrue(kwargs["load_soul_identity"])
        self.assertTrue(kwargs["skip_memory"])
        self.assertEqual(kwargs["platform"], "feishu")
        self.assertEqual(kwargs["user_id"], "ou_user")
        self.assertEqual(kwargs["chat_type"], "document_comment")
        self.assertEqual(kwargs["gateway_session_key"], "comment-doc:docx:token")

    @patch("gateway.session_context.clear_session_vars")
    @patch("gateway.session_context.set_session_vars", return_value=["token"])
    @patch("gateway.session_context.reset_session_vars")
    @patch("hermes_cli.profiles.get_active_profile_name", return_value="default")
    @patch("plugins.platforms.feishu.feishu_comment._run_comment_agent", return_value="done")
    def test_binds_feishu_identity_for_local_tools(
        self,
        mock_run,
        _mock_profile,
        mock_reset,
        mock_set,
        mock_clear,
    ):
        result = _run_comment_agent_with_context(
            "prompt",
            Mock(),
            session_key="comment-doc:docx:token",
            user_id="ou_user",
        )

        self.assertEqual(result, "done")
        mock_reset.assert_called_once_with()
        mock_set.assert_called_once_with(
            platform="feishu",
            chat_type="document_comment",
            user_id="ou_user",
            session_key="comment-doc:docx:token",
            profile="default",
            async_delivery=False,
            cron_session="",
        )
        mock_run.assert_called_once()
        mock_clear.assert_called_once_with(["token"])


class TestWikiReverseLookup(unittest.TestCase):
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @patch("plugins.platforms.feishu.feishu_comment._exec_request")
    def test_reverse_lookup_success(self, mock_exec):
        from plugins.platforms.feishu.feishu_comment import _reverse_lookup_wiki_token

        mock_exec.return_value = (0, "Success", {
            "node": {"node_token": "WIKI_TOKEN_123", "obj_token": "docx_abc"},
        })
        result = self._run(_reverse_lookup_wiki_token(Mock(), "docx", "docx_abc"))
        self.assertEqual(result, "WIKI_TOKEN_123")
        # Verify correct API params
        call_args = mock_exec.call_args
        queries = call_args[1].get("queries") or call_args[0][3]
        query_dict = dict(queries)
        self.assertEqual(query_dict["token"], "docx_abc")
        self.assertEqual(query_dict["obj_type"], "docx")


if __name__ == "__main__":
    unittest.main()
