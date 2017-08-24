#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import mrmProcessing.model.rawdata as rawdata
import mrmProcessing.model.sample as sample
import mrmProcessing.model.ionchrom as ionchrom
import mrmProcessing.model.batch as batch
import mrmProcessing.model.project as project

class testRawdata(unittest.TestCase):
    proj=project.project("projID")
    bat=batch.batch("batchID",proj)
    sam=sample.sample("sampID",bat,"file_name","sample_name")
    ioc=ionchrom.ionchrom("id")
    def test_001_init(self):
        """Tests if not having any parameters in the init call raises a TypeError"""
        self.assertRaises(TypeError,rawdata.rawdata)
    def test_002_init(self):
        """Tests if having only id parameter in the init call raises a TypeError"""
        self.assertRaises(TypeError,rawdata.rawdata,"id")
    def test_003_init(self):
        """Tests if having id and ionchrom parameter in the init call raises a TypeError"""
        self.assertRaises(TypeError,rawdata.rawdata,"id",testRawdata.ioc)
    def test_004_init(self):
        """Tests if passing the id, ionchrom and sample parameters generates a new object of type sample"""
        self.assertIsInstance(rawdata.rawdata("id",testRawdata.ioc,testRawdata.sam),rawdata.rawdata)

if __name__ == '__main__':
    unittest.main()