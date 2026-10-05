import ast
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_graph_links_and_30s_settings(self):
        for path in (ROOT / 'workflows').glob('*.json'):
            data=json.loads(path.read_text(encoding='utf-8'))
            nodes={n['id']:n for n in data['nodes']}
            links={l[0]:l for l in data['links']}
            self.assertEqual(len(links),len(data['links']))
            for id,a,slot,b,index,type in links.values():
                self.assertEqual(nodes[b]['inputs'][index]['link'],id)
                self.assertIn(id,nodes[a]['outputs'][slot]['links'])
                self.assertEqual(type,nodes[a]['outputs'][slot]['type'])
                self.assertEqual(type,nodes[b]['inputs'][index]['type'])
            for node in nodes.values():
                for inp in node['inputs']:
                    if inp['link'] is not None: self.assertIn(inp['link'],links)
                for out in node['outputs']:
                    for link in out.get('links') or []: self.assertIn(link,links)
            sampler=nodes[14]
            negative_link=next(i['link'] for i in sampler['inputs'] if i['name']=='negative')
            self.assertEqual(links[negative_link][1],10)
            self.assertEqual(nodes[7]['widgets_values']['force_rate'],16)
            self.assertEqual(nodes[7]['widgets_values']['frame_load_cap'],480)
            self.assertEqual(nodes[7]['widgets_values']['format'],'None')
            self.assertEqual(nodes[7]['widgets_values']['custom_width'],0)
            self.assertEqual(nodes[7]['widgets_values']['custom_height'],0)
            self.assertEqual(nodes[15]['widgets_values']['frame_rate'],16)
            self.assertEqual(sampler['widgets_values'][2:6],[16,30,81,17])
            self.assertEqual(sampler['widgets_values'][-1],'480p')
            self.assertFalse(any(n['type']=='ImageFromBatch' for n in nodes.values()))

    def test_custom_widget_layout_matches_python_schema(self):
        # Only introspect schemas; torch is unavailable in this CPU test environment.
        package=types.ModuleType('bernini_test_package')
        package.__path__=[str(ROOT)]
        spec=importlib.util.spec_from_file_location('bernini_test_package.nodes',ROOT/'nodes.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'bernini_test_package':package,'torch':types.ModuleType('torch')}):
            spec.loader.exec_module(module)
        # Original workflows can still call the node without the new optional widgets.
        with patch.object(module, 'load_config', side_effect=AssertionError('disabled node read config')):
            self.assertEqual(module.BerniniPromptEnhancerVLLM().run(
                'Original prompt', False, '', '', '', '', 0, 0.2, 512, 90), ('Original prompt',))
        scalar={'STRING','INT','FLOAT','BOOLEAN'}
        for path in (ROOT/'workflows').glob('*.json'):
            data=json.loads(path.read_text(encoding='utf-8'))
            for node in data['nodes']:
                if node['type'] not in module.NODE_CLASS_MAPPINGS: continue
                schema=module.NODE_CLASS_MAPPINGS[node['type']].INPUT_TYPES()
                widgets=[]
                for name, value in {**schema['required'], **schema.get('optional', {})}.items():
                    if isinstance(value[0],list) or value[0] in scalar:
                        widgets.append((name,value))
                    elif name in schema['required']:
                        self.assertIsNotNone(next(i['link'] for i in node['inputs'] if i['name']==name))
                self.assertEqual(len(node['widgets_values']),len(widgets))
                for (name,spec),value in zip(widgets,node['widgets_values']):
                    if isinstance(spec[0],list): self.assertIn(value,spec[0],name)

    def test_all_python_compiles(self):
        for path in ROOT.rglob('*.py'):
            compile(path.read_text(encoding='utf-8'),str(path),'exec')


if __name__=='__main__': unittest.main()
