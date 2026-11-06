from trace_replay import fetch_trace, fetch_traces, trace_to_run_result


class QueryClient:
    def query_json_rows(self, *, sql):
        assert "LIMIT 1" not in sql
        assert "invoke_agent %" in sql
        assert "agent run" in sql
        return {"rows": [
            {"trace_id": "abc", "all_messages": [
                {"role": "assistant", "parts": [{"type": "text", "content": "Final score"}]}
            ], "input_tokens": "300", "output_tokens": "30"},
            {"trace_id": "abc", "all_messages": [], "input_tokens": "200", "output_tokens": "20"},
            {"trace_id": "abc", "all_messages": [], "input_tokens": "100", "output_tokens": "10"},
        ]}


def test_fetch_grouped_trace_sums_all_runs():
    trace = fetch_trace("abc", QueryClient())
    result = trace_to_run_result(trace)
    assert result.usage().input_tokens == 600
    assert result.usage().output_tokens == 60
    assert result.output == "Final score"


def test_fetch_grouped_traces_sums_all_runs():
    trace = fetch_traces(["abc"], QueryClient())["abc"]
    assert trace.input_tokens == 600
    assert trace.output_tokens == 60
