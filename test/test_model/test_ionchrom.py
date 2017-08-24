#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import mrmProcessing.model.ionchrom as ionchrom

class testBatch(unittest.TestCase):
    def test_001_init(self):
        """Tests if not having any parameters in the init call raises a TypeError"""
        self.assertRaises(TypeError,ionchrom.ionchrom)
    def test_002_init(self):
        """Tests if passing the id parameter generates a new object of type ionchrom"""
        self.assertIsInstance(ionchrom.ionchrom("id"),ionchrom.ionchrom)
    
if __name__ == '__main__':
    unittest.main()