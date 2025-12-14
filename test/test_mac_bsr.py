"""
Unit tests for MAC layer BSR module (Phase 9).

Tests cover:
- BSR table encoding/decoding (5-bit and 8-bit tables)
- LCG mapping from QCI
- BSR report generation
- BSRManager integration with traffic layer
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

import unittest
import numpy as np
from collections import deque


class TestBSRTable(unittest.TestCase):
    """Test BSR table encoding and decoding."""

    def test_import(self):
        """Test module can be imported."""
        from mac import BSRTable, BSR_TABLE_5BIT, BSR_TABLE_8BIT
        self.assertIsNotNone(BSRTable)
        self.assertEqual(len(BSR_TABLE_5BIT), 32)  # 5-bit = 32 entries
        self.assertEqual(len(BSR_TABLE_8BIT), 256)  # 8-bit = 256 entries

    def test_5bit_table_values(self):
        """Test 5-bit BSR table boundary values per 3GPP TS 38.321."""
        from mac import BSR_TABLE_5BIT
        # Index 0 = 0 bytes
        self.assertEqual(BSR_TABLE_5BIT[0], 0)
        # Index 31 = >150,000 bytes
        self.assertGreaterEqual(BSR_TABLE_5BIT[31], 150000)

    def test_8bit_table_values(self):
        """Test 8-bit BSR table boundary values."""
        from mac import BSR_TABLE_8BIT
        # Index 0 = 0 bytes
        self.assertEqual(BSR_TABLE_8BIT[0], 0)
        # Index 255 should be max value
        self.assertGreater(BSR_TABLE_8BIT[255], 80000000)

    def test_bytes_to_index_5bit(self):
        """Test buffer size to BSR index conversion (5-bit)."""
        from mac import BSRTable
        table = BSRTable(bits=5)

        # 0 bytes -> index 0
        self.assertEqual(table.bytes_to_index(0), 0)

        # Small values
        self.assertEqual(table.bytes_to_index(5), 0)
        self.assertEqual(table.bytes_to_index(10), 1)
        self.assertEqual(table.bytes_to_index(15), 2)

        # Large values -> index 31
        self.assertEqual(table.bytes_to_index(200000), 31)

    def test_bytes_to_index_8bit(self):
        """Test buffer size to BSR index conversion (8-bit)."""
        from mac import BSRTable
        table = BSRTable(bits=8)

        # 0 bytes -> index 0
        self.assertEqual(table.bytes_to_index(0), 0)

        # Large values -> high indices
        idx = table.bytes_to_index(1000000)
        self.assertGreater(idx, 100)

    def test_index_to_bytes(self):
        """Test BSR index to buffer size conversion."""
        from mac import BSRTable
        table = BSRTable(bits=8)

        # Index 0 = 0 bytes
        self.assertEqual(table.index_to_bytes(0), 0)

        # Clamp to valid range
        self.assertEqual(table.index_to_bytes(-1), 0)
        self.assertGreater(table.index_to_bytes(255), 0)

    def test_roundtrip_conversion(self):
        """Test bytes -> index -> bytes roundtrip is consistent."""
        from mac import BSRTable
        table = BSRTable(bits=8)

        # Small value
        original = 100
        idx = table.bytes_to_index(original)
        recovered = table.index_to_bytes(idx)
        # Recovered should be <= original (quantization)
        self.assertLessEqual(recovered, original)

    def test_invalid_bits_raises(self):
        """Test invalid bit count raises ValueError."""
        from mac import BSRTable
        with self.assertRaises(ValueError):
            BSRTable(bits=6)


class TestLCGMapping(unittest.TestCase):
    """Test Logical Channel Group mapping."""

    def test_qci_to_lcg_default(self):
        """Test default QCI to LCG mapping."""
        from mac import qci_to_lcg

        # Signaling
        self.assertEqual(qci_to_lcg(5), 0)

        # Voice
        self.assertEqual(qci_to_lcg(1), 1)

        # Video
        self.assertEqual(qci_to_lcg(2), 2)
        self.assertEqual(qci_to_lcg(4), 2)

        # URLLC
        self.assertEqual(qci_to_lcg(80), 3)
        self.assertEqual(qci_to_lcg(85), 3)

        # Best effort
        self.assertEqual(qci_to_lcg(9), 7)
        self.assertEqual(qci_to_lcg(6), 7)

    def test_lcg_manager(self):
        """Test LCGManager functionality."""
        from mac import LCGManager

        mgr = LCGManager()

        # Test priority lookup
        self.assertEqual(mgr.get_priority(0), 1)  # Signaling - high priority
        self.assertEqual(mgr.get_priority(7), 7)  # Best effort - lowest

        # Test highest priority LCG selection
        self.assertEqual(mgr.get_highest_priority_lcg([7, 3, 1]), 3)  # URLLC has priority 0

    def test_custom_mapping(self):
        """Test custom QCI to LCG mapping."""
        from mac import LCGManager

        custom = {99: 5}  # Custom QCI 99 -> LCG 5
        mgr = LCGManager(custom_mapping=custom)

        self.assertEqual(mgr.get_lcg(99), 5)
        self.assertEqual(mgr.get_lcg(9), 7)  # Default still works


class TestBSRReport(unittest.TestCase):
    """Test BSR report data structure."""

    def test_bsr_report_creation(self):
        """Test BSRReport creation and properties."""
        from mac import BSRReport, BSRTrigger

        report = BSRReport(
            ue_id=0,
            lcg_id=1,
            buffer_size_bytes=1000,
            bsr_index=15,
            timestamp_tti=100,
            trigger=BSRTrigger.REGULAR,
        )

        self.assertEqual(report.ue_id, 0)
        self.assertEqual(report.lcg_id, 1)
        self.assertEqual(report.buffer_size_bytes, 1000)
        self.assertFalse(report.is_empty)

    def test_empty_report(self):
        """Test empty BSR report detection."""
        from mac import BSRReport, BSRTrigger

        report = BSRReport(
            ue_id=0,
            lcg_id=7,
            buffer_size_bytes=0,
            bsr_index=0,
            timestamp_tti=0,
            trigger=BSRTrigger.PERIODIC,
        )

        self.assertTrue(report.is_empty)


class TestBSRManager(unittest.TestCase):
    """Test BSRManager integration."""

    def test_manager_creation(self):
        """Test BSRManager creation."""
        from mac import BSRManager

        mgr = BSRManager(n_ue=10, config={"tti_ms": 1.0})
        self.assertEqual(mgr.n_ue, 10)

    def test_get_pending_data(self):
        """Test pending data retrieval."""
        from mac import BSRManager

        mgr = BSRManager(n_ue=5)

        # Initially empty
        self.assertEqual(mgr.get_pending_data_bytes(0), 0)

        # All pending data array
        pending = mgr.get_all_pending_data()
        self.assertEqual(len(pending), 5)
        self.assertTrue(np.all(pending == 0))

    def test_non_empty_ues(self):
        """Test non-empty UE list."""
        from mac import BSRManager

        mgr = BSRManager(n_ue=5)

        # Initially no UEs with data
        self.assertEqual(mgr.get_non_empty_ues(), [])

    def test_dequeue_bytes(self):
        """Test dequeue notification."""
        from mac import BSRManager

        mgr = BSRManager(n_ue=5)

        # Should not raise even if buffer empty
        mgr.dequeue_bytes(0, 1000)

    def test_statistics(self):
        """Test statistics retrieval."""
        from mac import BSRManager

        mgr = BSRManager(n_ue=10)
        stats = mgr.get_statistics()

        self.assertEqual(stats["total_ues"], 10)
        self.assertEqual(stats["non_empty_ues"], 0)
        self.assertEqual(stats["total_buffer_bytes"], 0)

    def test_reset(self):
        """Test manager reset."""
        from mac import BSRManager

        mgr = BSRManager(n_ue=5)
        mgr.reset()

        self.assertEqual(mgr.get_pending_data_bytes(0), 0)


class TestBSRWithTraffic(unittest.TestCase):
    """Test BSR integration with Traffic layer."""

    def test_update_from_buffer_manager(self):
        """Test BSR update from traffic buffer manager."""
        from mac import BSRManager
        from traffic import BufferManager, Packet

        # Create traffic buffer with some packets
        buffer_mgr = BufferManager(n_ue=3)

        # Add packets to UE 0
        pkt1 = Packet(
            packet_id=0, ue_id=0, bearer_id=0,
            size_bits=8000, arrival_tti=0,
            deadline_tti=100, priority=5, qci=9
        )
        buffer_mgr.enqueue(pkt1)

        pkt2 = Packet(
            packet_id=1, ue_id=0, bearer_id=0,
            size_bits=4000, arrival_tti=0,
            deadline_tti=100, priority=5, qci=9
        )
        buffer_mgr.enqueue(pkt2)

        # Create BSR manager and update
        bsr_mgr = BSRManager(n_ue=3, config={"tti_ms": 1.0})
        reports = bsr_mgr.update_from_buffer_manager(buffer_mgr, current_tti=0)

        # Should have generated BSR report for UE 0
        self.assertGreater(len(reports), 0)

        # UE 0 should have pending data
        pending = bsr_mgr.get_pending_data_bytes(0)
        self.assertGreater(pending, 0)

        # UE 0 should be in non-empty list
        self.assertIn(0, bsr_mgr.get_non_empty_ues())

    def test_lcg_separation(self):
        """Test packets with different QCI map to different LCGs."""
        from mac import BSRManager
        from traffic import BufferManager, Packet

        buffer_mgr = BufferManager(n_ue=1)

        # Voice packet (QCI 1 -> LCG 1)
        pkt_voice = Packet(
            packet_id=0, ue_id=0, bearer_id=0,
            size_bits=8000, arrival_tti=0,
            deadline_tti=100, priority=1, qci=1
        )
        buffer_mgr.enqueue(pkt_voice)

        # Best effort packet (QCI 9 -> LCG 7)
        pkt_be = Packet(
            packet_id=1, ue_id=0, bearer_id=1,
            size_bits=8000, arrival_tti=0,
            deadline_tti=300, priority=9, qci=9
        )
        buffer_mgr.enqueue(pkt_be)

        # Update BSR
        bsr_mgr = BSRManager(n_ue=1, config={"tti_ms": 1.0})
        reports = bsr_mgr.update_from_buffer_manager(buffer_mgr, current_tti=0)

        # Should have reports for both LCGs
        lcgs_reported = {r.lcg_id for r in reports}
        self.assertIn(1, lcgs_reported)  # Voice LCG
        self.assertIn(7, lcgs_reported)  # Best effort LCG

    def test_bsr_disabled(self):
        """Test BSR can be disabled."""
        from mac import BSRManager
        from traffic import BufferManager, Packet

        buffer_mgr = BufferManager(n_ue=1)
        pkt = Packet(
            packet_id=0, ue_id=0, bearer_id=0,
            size_bits=8000, arrival_tti=0,
            deadline_tti=100, priority=5, qci=9
        )
        buffer_mgr.enqueue(pkt)

        # BSR disabled
        bsr_mgr = BSRManager(n_ue=1, config={"enable_bsr": False})
        reports = bsr_mgr.update_from_buffer_manager(buffer_mgr, current_tti=0)

        # Should return empty list when disabled
        self.assertEqual(len(reports), 0)


class TestConfigIntegration(unittest.TestCase):
    """Test MAC configuration integration."""

    def test_mac_config_exists(self):
        """Test MACConfig exists in schema."""
        from config.schema import MACConfig, NTNSimConfig

        cfg = MACConfig()
        self.assertTrue(cfg.enable_bsr)
        self.assertEqual(cfg.bsr_table_bits, 8)

    def test_flat_dict_roundtrip(self):
        """Test MAC config in flat dict conversion."""
        from config.schema import NTNSimConfig

        cfg = NTNSimConfig()
        cfg.mac.enable_bsr = False
        cfg.mac.bsr_table_bits = 5

        flat = cfg.to_flat_dict()
        self.assertFalse(flat["enable_bsr"])
        self.assertEqual(flat["bsr_table_bits"], 5)

        # Roundtrip
        cfg2 = NTNSimConfig.from_flat_dict(flat)
        self.assertFalse(cfg2.mac.enable_bsr)
        self.assertEqual(cfg2.mac.bsr_table_bits, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
