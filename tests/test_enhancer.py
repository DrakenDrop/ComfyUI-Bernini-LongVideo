import io
import base64
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError, URLError

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from enhancer import enhance, endpoint, image_parts


class EnhancerTests(unittest.TestCase):
    def call(self, **kwargs):
        values = dict(instruction='Ubah baju menjadi biru',scene_description='',
            base_url='http://localhost:8000/v1', model='test', api_key_env='TEST_VLLM_KEY',
            temperature=0.2,max_tokens=512,timeout=10)
        values.update(kwargs)
        return enhance(**values)

    def response(self, value):
        response=MagicMock()
        response.__enter__.return_value.read.return_value=json.dumps(value).encode()
        return response

    def test_endpoint(self):
        self.assertEqual(endpoint('http://localhost:8000/'),'http://localhost:8000/v1/chat/completions')
        self.assertEqual(endpoint('https://host/proxy/v1/'),'https://host/proxy/v1/chat/completions')
        for url in ['file:///x','http://u:p@host','https://host?q=x','https://host/#x']:
            with self.assertRaises(ValueError): endpoint(url)

    @patch('enhancer.build_opener')
    def test_request_and_secret_not_in_payload(self, opener):
        opener.return_value.open.return_value=self.response({'choices':[{'message':{'content':'Edit shirt to blue.'},'finish_reason':'stop'}]})
        with patch.dict(os.environ, {'TEST_VLLM_KEY':'test-secret'}):
            self.assertEqual(self.call(),'Edit shirt to blue.')
        req=opener.return_value.open.call_args.args[0]
        self.assertEqual(req.headers['Authorization'],'Bearer test-secret')
        self.assertNotIn('test-secret',req.data.decode())
        body=json.loads(req.data)
        self.assertIsInstance(body['messages'][1]['content'],str)
        self.assertEqual(body['model'],'test')

    def test_images_bounded_and_data_uri(self):
        frames=np.zeros((10,20,30,3),dtype=np.float32)
        parts=image_parts(frames,4)
        self.assertEqual(len(parts),4)
        self.assertTrue(parts[0]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(image_parts(frames,0),[])

    @patch('enhancer.build_opener')
    def test_multimodal_request(self,opener):
        opener.return_value.open.return_value=self.response({'choices':[{'message':{'content':'Blue shirt.'}}]})
        self.call(frames=np.zeros((5,20,30,3),dtype=np.float32),sample_frames=4)
        body=json.loads(opener.return_value.open.call_args.args[0].data)
        content = body['messages'][1]['content']
        self.assertEqual(len([p for p in content if p['type'] == 'image_url']), 4)
        self.assertIn('SOURCE FRAMES', content[1]['text'])

    @patch('enhancer.build_opener')
    def test_source_and_reference_have_distinct_roles_and_pixels(self, opener):
        opener.return_value.open.return_value=self.response({'choices':[{'message':{'content':'Match the reference shirt.'}}]})
        self.call(frames=np.zeros((5,20,30,3),dtype=np.float32), sample_frames=1,
                  reference_images=np.ones((3,20,30,3),dtype=np.float32))
        content=json.loads(opener.return_value.open.call_args.args[0].data)['messages'][1]['content']
        self.assertEqual([p['type'] for p in content], ['text','text','image_url','text','image_url'])
        self.assertIn('SOURCE FRAMES',content[1]['text'])
        self.assertIn('REFERENCE IMAGE',content[3]['text'])
        def pixels(part):
            return np.asarray(Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',')[1]))))
        self.assertLess(pixels(content[2]).mean(), 1)
        self.assertGreater(pixels(content[4]).mean(), 254)

    @patch('enhancer.build_opener')
    def test_reference_only_is_vision_even_with_zero_source_samples(self, opener):
        opener.return_value.open.return_value=self.response({'choices':[{'message':{'content':'Use the reference shirt.'}}]})
        self.call(sample_frames=0, reference_images=np.ones((1,20,30,3),dtype=np.float32))
        content=json.loads(opener.return_value.open.call_args.args[0].data)['messages'][1]['content']
        self.assertEqual(len(content),3)
        self.assertIn('REFERENCE IMAGE',content[1]['text'])

    @patch('enhancer.build_opener')
    def test_empty_truncated_and_malformed_rejected(self,opener):
        for data in [{}, {'choices':[]}, {'choices':[{'message':{'content':''}}]},
                     {'choices':[{'message':{'content':'partial'},'finish_reason':'length'}]}]:
            opener.return_value.open.return_value=self.response(data)
            with self.assertRaises(RuntimeError): self.call()

    @patch('enhancer.build_opener')
    def test_errors_do_not_echo_server_secrets(self,opener):
        for exc in [HTTPError('https://secret',401,'secret',{},io.BytesIO(b'secret')),
                    URLError('secret'),TimeoutError('secret')]:
            opener.return_value.open.side_effect=exc
            with self.assertRaises(RuntimeError) as ctx: self.call()
            self.assertNotIn('secret',str(ctx.exception))

    @patch('enhancer.build_opener')
    def test_invalid_input_does_not_send(self,opener):
        with self.assertRaises(ValueError): self.call(instruction='')
        opener.assert_not_called()

    @patch('enhancer.build_opener')
    def test_auto_model_get_then_post(self, opener):
        for model in ('auto', '', ' AUTO '):
            opener.reset_mock()
            opener.return_value.open.side_effect = [
                self.response({'data': [{'id': 'served-name'}, {'id': 'served-name'}]}),
                self.response({'choices': [{'message': {'content': 'Blue shirt.'}}]})]
            with patch.dict(os.environ, {'TEST_VLLM_KEY': 'secret'}):
                self.assertEqual(self.call(model=model), 'Blue shirt.')
            discovery, completion = [call.args[0] for call in opener.return_value.open.call_args_list]
            self.assertEqual(discovery.get_method(), 'GET')
            self.assertEqual(discovery.full_url, 'http://localhost:8000/v1/models')
            self.assertEqual(discovery.headers['Authorization'], 'Bearer secret')
            self.assertEqual(json.loads(completion.data)['model'], 'served-name')

    @patch('enhancer.build_opener')
    def test_auto_model_rejects_empty_ambiguous_and_invalid_lists(self, opener):
        for data in ({}, [], {'data': []}, {'data': [{'id': 1}]},
                     {'data': [{'id': 'one'}, {'id': 'two'}]}):
            opener.reset_mock()
            opener.return_value.open.return_value = self.response(data)
            with self.assertRaises(RuntimeError): self.call(model='auto')
            self.assertEqual(opener.return_value.open.call_count, 1)


if __name__=='__main__': unittest.main()
