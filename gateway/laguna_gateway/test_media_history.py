import unittest
from copy import deepcopy
from media_history import compact_image_history


def count(body):
    def walk(v):
        if isinstance(v, list): return sum(map(walk, v))
        if isinstance(v, dict):
            return int(v.get('type') == 'input_image') + sum(walk(x) for x in v.values())
        return 0
    return {'image':walk(body.get('input', []))}


def latest(items):
    return next((i for i in reversed(range(len(items))) if items[i].get('role')=='user'),None)


def user(n):
    return {'type':'message','role':'user','content':[{'type':'input_image','image_url':'data:original-'+str(i)} for i in range(n)]}


class Tests(unittest.IsolatedAsyncioTestCase):
    async def call(self, body):
        async def summarize(items): return 'older completed work',1
        return await compact_image_history(body,32,count,latest,summarize,lambda s:{'role':'developer','content':s})

    async def test_twelve_original_images_unchanged(self):
        body={'input':[user(12)]};original=deepcopy(body)
        out,n,_=await self.call(body)
        self.assertEqual(out,original);self.assertEqual(n,0)

    async def test_compacts_older_turn_retains_all_current_originals(self):
        live=[user(12),{'type':'function_call','call_id':'new'},{'type':'function_call_output','call_id':'new','output':[{'type':'input_image','image_url':'data:tool-original'}]}]
        body={'input':[{'role':'system','content':'policy'},user(21),{'role':'assistant','content':'done'},*live]}
        original=deepcopy(body);out,n,_=await self.call(body)
        self.assertGreater(n,0);self.assertEqual(out['input'][-len(live):],live)
        self.assertEqual(out['input'][0],body['input'][0]);self.assertEqual(body,original)
        self.assertEqual(count(out)['image'],13)

    async def test_never_summarizes_unseen_or_live_overflow(self):
        body={'input':[user(33)]};out,n,_=await self.call(body)
        self.assertEqual(out,body);self.assertEqual(n,0)
        body={'input':[user(25),user(12)]};out,n,_=await self.call(body)
        self.assertEqual(out,body);self.assertEqual(n,0)

    async def test_cross_boundary_tool_call_protected(self):
        body={'input':[user(25),{'type':'function_call','call_id':'open'},user(12),{'type':'function_call_output','call_id':'open','output':'ok'}]}
        out,n,_=await self.call(body)
        self.assertEqual(out,body);self.assertEqual(n,0)


if __name__=='__main__':unittest.main()
