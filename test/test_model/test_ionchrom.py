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
    # def test_003_init(self):
    #     """Tests if having both the id and project parameter generates a new object of type batch"""
    #     self.assertIsInstance(ionchrom.ionchrom("id",project.project("projID")),batch.batch)
    # def test_004_init(self):
    #     """Tests if project parameter of the batch class returns the right name"""
    #     proj=project.project("projID")
    #     bat=batch.batch("id",proj)
    #     self.assertTrue(bat.project.myid==proj.myid)

if __name__ == '__main__':
    unittest.main()