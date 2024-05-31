import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from agent import Agent, AgentFactory


class TestModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.output_size = 2
        self.model_cls = TestModel
        self.arguments = {
            "input_size": self.input_size,
            "output_size": self.output_size,
        }

    def test_agent_initialization(self):
        agent = Agent(self.model_cls, self.arguments)
        self.assertIsInstance(agent, Agent)
        self.assertIsInstance(agent.model, TestModel)
        self.assertEqual(agent.arguments, self.arguments)
        self.assertIsInstance(agent.id, str)

    def test_agent_forward(self):
        agent = Agent(self.model_cls, self.arguments)
        input_tensor = torch.randn(1, self.input_size)
        output_tensor = agent(input_tensor)
        self.assertEqual(output_tensor.shape, (1, self.output_size))

    def test_agent_initialization_type_error(self):
        wrong_arguments = {"wrong_arg": 10}
        with self.assertRaises(ValueError) as context:
            Agent(self.model_cls, wrong_arguments)
        self.assertIn("Error initializing model with arguments", str(context.exception))

    def test_agent_initialization_unexpected_error(self):
        # Mock the model class to raise an unexpected error
        def mock_model_init(*args, **kwargs):
            raise RuntimeError("Unexpected error")

        mock_model_cls = MagicMock(side_effect=mock_model_init)
        with self.assertRaises(ValueError) as context:
            Agent(mock_model_cls, self.arguments)
        self.assertIn("Unexpected error initializing model", str(context.exception))


class TestAgentFactory(unittest.TestCase):
    def setUp(self):
        self.model_cls = TestModel
        self.factory = AgentFactory(self.model_cls)
        self.arguments = {"input_size": 10, "output_size": 2}

    def test_factory_initialization(self):
        self.assertIsInstance(self.factory, AgentFactory)
        self.assertEqual(self.factory.model_cls, self.model_cls)

    def test_factory_create_agent(self):
        agent = self.factory.create(self.arguments)
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.arguments, self.arguments)

    def test_factory_create_agent_error(self):
        wrong_arguments = {"wrong_arg": 10}
        with self.assertRaises(ValueError):
            self.factory.create(wrong_arguments)


if __name__ == "__main__":
    unittest.main()
