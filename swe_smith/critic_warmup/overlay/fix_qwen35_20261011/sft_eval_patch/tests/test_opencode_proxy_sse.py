"""Exercise the real gateway handler over HTTP without a model or testbed."""
import ast
import http.client
import http.server
import json
import logging
from pathlib import Path
import socketserver
import threading
import urllib.parse

import pytest

PATH = Path(__file__).resolve().parents[3] / 'examples/swe_smith/agents/opencode_agent.py'


@pytest.mark.parametrize('reason', ['stop', 'tool_calls', 'length'])
@pytest.mark.parametrize('stream', [False, True])
def test_proxy_preserves_completion_contract(reason, stream):
    names = {'_clip_text', '_message_text', '_set_message_text', '_is_tool_message',
             'truncate_openai_body', 'completion_to_sse', '_GatewayProxyHandler', '_ReusableTCPServer'}
    tree = ast.parse(PATH.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    ns = dict(http=http, json=json, logging=logging, socketserver=socketserver,
              urllib=urllib, Any=object, TOOL_CHAR_LIMIT=16000, OUTPUT_LIMIT=6144,
              SAMPLE_TIMEOUT=10, log=logging.getLogger('test-proxy'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), ns)
    message = {'role': 'assistant', 'content': 'hello'}
    if reason == 'tool_calls':
        message = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call1', 'type': 'function', 'function': {'name': 'edit', 'arguments': '{"text":"  x\\n"}'}}]}
    result = {'id': 'test', 'created': 1, 'model': 'test', 'choices': [
        {'index': 0, 'message': message, 'finish_reason': reason}], 'usage': {'total_tokens': 7}}
    requests = []

    class Upstream(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    with http.server.ThreadingHTTPServer(('127.0.0.1', 0), Upstream) as upstream:
        handler = ns['_GatewayProxyHandler']
        handler.agl_port = upstream.server_port
        with ns['_ReusableTCPServer'](('127.0.0.1', 0), handler) as proxy:
            threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, proxy)]
            for thread in threads:
                thread.start()
            conn = http.client.HTTPConnection('127.0.0.1', proxy.server_address[1], timeout=10)
            try:
                conn.request('POST', '/v1/chat/completions', json.dumps({'messages': [], 'stream': stream}),
                             {'Content-Type': 'application/json'})
                response = conn.getresponse()
                raw = response.read().decode()
                assert response.status == 200
                assert requests[0]['stream'] is False
                if stream:
                    assert response.getheader('Content-Type').startswith('text/event-stream')
                    assert raw.endswith('data: [DONE]\n\n')
                    chunks = [json.loads(x[6:]) for x in raw.split('\n\n') if x.startswith('data: ') and x != 'data: [DONE]']
                    assert chunks[1]['choices'][0]['finish_reason'] == reason
                    delta = chunks[0]['choices'][0]['delta']
                    assert delta['content'] == message['content']
                    if reason == 'tool_calls':
                        assert delta['tool_calls'][0]['index'] == 0
                        assert delta['tool_calls'][0]['function'] == message['tool_calls'][0]['function']
                else:
                    assert json.loads(raw) == result
            finally:
                conn.close()
                proxy.shutdown()
                upstream.shutdown()
                for thread in threads:
                    thread.join(timeout=5)
