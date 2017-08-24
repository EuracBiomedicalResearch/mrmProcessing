#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import mrmProcessing.model.project as project
import mrmProcessing.model.batch as batch


class testBatch(unittest.TestCase):
    def test_init01(self):
        """Tests if not having any parameters in the init call raises a TypeError"""
        self.assertRaises(TypeError,batch.batch)
    def test_init02(self):
        """Tests if having only the id parameter raises a TypeError"""
        self.assertRaises(TypeError,batch.batch,"id")
    def test_init03(self):
        """Tests if having both the id and project parameter generates a new object of type batch"""
        self.assertIsInstance(batch.batch("id",project.project("projID")),batch.batch)
        """Tests if project parameter of the batch class returns the right name"""
        proj=project.project("projID")
        bat=batch.batch("id",proj)
        self.assertTrue(bat.project.myid==proj.myid)

if __name__ == '__main__':
    unittest.main()