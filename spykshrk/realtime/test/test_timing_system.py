from unittest import TestCase
from spykshrk.realtime.timing_system import TimingMessage

class TestTimingMessage(TestCase):

    def setUp(self):
        self.msg = TimingMessage(label='Test', timestamp=100, start_rank=1)
        self.msg.record_time(rank=2)
        self.msg.record_time(rank=3)

    def test_TimingMessage_message_len(self):

        msg_bytes = self.msg.pack()
        msg_unpack = TimingMessage.unpack(message_bytes=msg_bytes, message_len=len(msg_bytes))

        self.assertEqual(self.msg.label, msg_unpack.label,
                         'Serialization failed, labels do not match ({}, {})'.
                         format(self.msg.label, msg_unpack.label))

        self.assertEqual(self.msg.timestamp, msg_unpack.timestamp,
                         'Serialization failed, timestamps do not match ({}, {})'.
                         format(self.msg.timestamp, msg_unpack.timestamp))

        self.assertEqual(self.msg.timing_data, msg_unpack.timing_data,
                         'Serialization failed, timing_data does not match ({}, {})'.
                         format(self.msg.timing_data, msg_unpack.timing_data))

    def test_TimingMessage(self):

        msg_bytes = self.msg.pack()
        msg_unpack = TimingMessage.unpack(message_bytes=msg_bytes)

        self.assertEqual(self.msg.label, msg_unpack.label,
                         'Serialization failed, labels do not match ({}, {})'.
                         format(self.msg.label, msg_unpack.label))

        self.assertEqual(self.msg.timestamp, msg_unpack.timestamp,
                         'Serialization failed, timestamps do not match ({}, {})'.
                         format(self.msg.timestamp, msg_unpack.timestamp))

        self.assertEqual(self.msg.timing_data, msg_unpack.timing_data,
                         'Serialization failed, timing_data does not match ({}, {})'.
                         format(self.msg.timing_data, msg_unpack.timing_data))
