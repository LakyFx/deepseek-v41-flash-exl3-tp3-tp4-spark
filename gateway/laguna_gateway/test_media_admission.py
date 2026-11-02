import asyncio
import os
import unittest
from types import SimpleNamespace
from media_admission import media_request, media_close

IMAGE = {'input': [{'type': 'input_image', 'image_url': 'data:image/png;base64,AA'}]}


class Engine:
    def __init__(self):
        self.started = []
        self.finish = asyncio.Event()

    @media_request()
    async def complete(self, body, retry=False):
        if retry:
            return await self.complete(body)
        self.started.append(body)
        if body:
            await self.finish.wait()
        return 'ok'

    @media_request(stream=True, close_method='close')
    async def begin(self, body):
        self.started.append(body)
        await self.finish.wait()
        return SimpleNamespace(media_release=None)

    @media_close
    async def close(self, handle, **kwargs):
        pass


class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        os.environ['DGX_MEDIA_SERIAL_REQUESTS'] = '1'

    async def test_queue_text_bypass_and_cancelled_waiter(self):
        e = Engine()
        first = asyncio.create_task(e.complete(IMAGE, retry=True))
        await asyncio.sleep(.02)
        second = asyncio.create_task(e.complete(IMAGE))
        await asyncio.sleep(.02)
        self.assertEqual(len(e.started), 1)
        self.assertEqual(await e.complete({}), 'ok')
        second.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await second
        e.finish.set()
        await first
        self.assertEqual(await e.complete(IMAGE), 'ok')

    async def test_stream_holds_until_cleanup(self):
        e = Engine(); e.finish.set()
        handle = await e.begin(IMAGE)
        waiter = asyncio.create_task(e.complete(IMAGE))
        await asyncio.sleep(.02)
        self.assertFalse(waiter.done())
        await e.close(handle)
        await e.close(handle)
        self.assertEqual(await waiter, 'ok')
        self.assertEqual(e._media_request_gate._value, 1)

    async def test_cancel_running_nonstream_keeps_bound(self):
        e = Engine()
        first = asyncio.create_task(e.complete(IMAGE))
        await asyncio.sleep(.02)
        first.cancel()
        second = asyncio.create_task(e.complete(IMAGE))
        await asyncio.sleep(.02)
        self.assertEqual(len(e.started), 1)
        e.finish.set()
        with self.assertRaises(asyncio.CancelledError): await first
        await second
        self.assertEqual(e._media_request_gate._value, 1)

    async def test_cancel_stream_start_closes_and_releases(self):
        e = Engine()
        first = asyncio.create_task(e.begin(IMAGE))
        await asyncio.sleep(.02)
        first.cancel()
        e.finish.set()
        with self.assertRaises(asyncio.CancelledError): await first
        self.assertEqual(e._media_request_gate._value, 1)

    async def test_backend_exception_releases(self):
        class Bad(Engine):
            @media_request()
            async def fail(self, body): raise ValueError('upstream failure')
        e = Bad()
        with self.assertRaises(ValueError): await e.fail(IMAGE)
        self.assertEqual(e._media_request_gate._value, 1)


if __name__ == '__main__': unittest.main()
