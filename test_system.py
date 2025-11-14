"""
Simple test script to verify the ML Agent System works correctly.

This script performs basic import tests and validates core functionality.
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from ml_agent_system import MLOrchestrator, ProblemType
        print("✓ Main module imports successful")

        from ml_agent_system.agents import (
            ResearchAgent, DataDesignAgent, DataGenerationAgent,
            ModelBuilderAgent, EvaluationAgent
        )
        print("✓ Agent imports successful")

        from ml_agent_system.core import (
            BaseAgent, get_config, SystemConfig
        )
        print("✓ Core module imports successful")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without requiring API keys."""
    print("\nTesting basic functionality...")

    try:
        from ml_agent_system import MLOrchestrator, ProblemType
        from ml_agent_system.core import SystemConfig, set_config

        # Set test configuration
        config = SystemConfig(
            log_level="WARNING",  # Reduce noise
            output_dir="./test_outputs",
            data_dir="./test_data",
            reports_dir="./test_reports"
        )
        set_config(config)
        print("✓ Configuration system works")

        # Initialize orchestrator
        orchestrator = MLOrchestrator()
        print("✓ Orchestrator initialization successful")

        # Test agent status
        status = orchestrator.get_agent_status()
        assert len(status) == 5, "Should have 5 agents"
        print("✓ Agent status retrieval works")

        # Test individual agent initialization
        from ml_agent_system.agents import ResearchAgent
        agent = ResearchAgent()
        print("✓ Individual agent initialization works")

        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_models():
    """Test data models."""
    print("\nTesting data models...")

    try:
        from ml_agent_system.core.models import (
            ProblemType, MLTask, DataSpecification
        )

        # Test ProblemType enum
        assert ProblemType.CLASSIFICATION.value == "classification"
        print("✓ ProblemType enum works")

        # Test MLTask model
        task = MLTask(
            description="Test problem",
            problem_type=ProblemType.CLASSIFICATION
        )
        assert task.description == "Test problem"
        print("✓ MLTask model works")

        # Test DataSpecification
        spec = DataSpecification(
            num_samples=100,
            features=[{"name": "test", "type": "numerical"}],
            target={"name": "target", "type": "categorical"}
        )
        assert spec.num_samples == 100
        print("✓ DataSpecification model works")

        return True
    except Exception as e:
        print(f"✗ Data model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("ML Agent System - Test Suite")
    print("="*80 + "\n")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Data Models", test_data_models()))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Add your API keys to .env file")
        print("2. Run examples: python ml_agent_system/examples/basic_example.py")
        print("="*80 + "\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
