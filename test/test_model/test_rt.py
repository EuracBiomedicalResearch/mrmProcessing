#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import mrmProcessing.model.rt as rt
import mrmProcessing.model.ionchrom as ionchrom
import mrmProcessing.model.batch as batch
import mrmProcessing.model.project as project

class testRt(unittest.TestCase):
    proj=project.project("projID")
    bat=batch.batch("batchID",proj)
    ionch=ionchrom.ionchrom("ioncID")
    def test_001_init(self):
        """Tests if not having any parameters in the init call raises a TypeError"""
        self.assertRaises(TypeError,rt.rt)
    def test_002_init(self):
        """Tests if having only id parameter in the init call raises a TypeError"""
        self.assertRaises(TypeError,rt.rt,"id")
    def test_003_init(self):
        """Tests if having id and ionchrom parameter in the init call raises a TypeError"""
        self.assertRaises(TypeError,rt.rt,"id",testRt.ionch)
    def test_004_init(self):
        """Tests if passing the id, ionchrom and batch parameters generates a new object of type sample"""
        self.assertIsInstance(rt.rt("id",testRt.ionch,testRt.bat),rt.rt)

if __name__ == '__main__':
    unittest.main()