#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import model.project as project


class testProject(unittest.TestCase):
    def test_init(self):
        assertIsNotInstance(project.project(),project.project)

if __name__ == '__main__':
    unittest.main()