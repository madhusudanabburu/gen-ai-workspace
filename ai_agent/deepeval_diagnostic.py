#!/usr/bin/env python3
"""
Comprehensive DeepEval Diagnostic Script
Identifies the root cause of 0.0 scores issue
"""

import os
import sys
import traceback
from datetime import datetime

def load_env():
    """Load environment variables"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return True
    except ImportError:
        print("Warning: python-dotenv not available")
        return False

def test_openai_api():
    """Test if OpenAI API is working directly"""
    print("\n" + "="*60)
    print("🔥 TESTING OPENAI API DIRECTLY")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        import openai
        
        # Test with the new OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        print("🔄 Testing OpenAI API with simple completion...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello, this is a test'"}
            ],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"✅ OpenAI API works! Response: {result}")
        print(f"   Usage: {response.usage}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        traceback.print_exc()
        return False

def test_deepeval_imports():
    """Test DeepEval imports and version"""
    print("\n" + "="*60)
    print("📦 TESTING DEEPEVAL IMPORTS")
    print("="*60)
    
    try:
        import deepeval
        print(f"✅ DeepEval imported successfully")
        print(f"   Version: {getattr(deepeval, '__version__', 'Unknown')}")
        
        from deepeval import evaluate, assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric
        
        print("✅ Core DeepEval components imported")
        return True
        
    except Exception as e:
        print(f"❌ DeepEval import failed: {e}")
        traceback.print_exc()
        return False

def test_simple_deepeval():
    """Test the simplest possible DeepEval evaluation"""
    print("\n" + "="*60)
    print("🧪 TESTING SIMPLE DEEPEVAL EVALUATION")
    print("="*60)
    
    try:
        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric
        
        # Create the simplest possible test case
        test_case = LLMTestCase(
            input="What is 2+2?",
            actual_output="2+2 equals 4"
        )
        
        # Create a single metric with very low threshold
        metric = AnswerRelevancyMetric(threshold=0.1)
        
        print("🔄 Running evaluation...")
        print(f"   Input: {test_case.input}")
        print(f"   Output: {test_case.actual_output}")
        print(f"   Metric: {metric.__class__.__name__} (threshold={metric.threshold})")
        
        # Run evaluation
        result = evaluate([test_case], [metric])
        
        print(f"✅ Evaluation completed!")
        print(f"   Result type: {type(result)}")
        print(f"   Result: {result}")
        
        # Detailed inspection of the result
        print(f"\n📋 DETAILED RESULT INSPECTION:")
        attrs = [attr for attr in dir(result) if not attr.startswith('_')]
        print(f"   Available attributes: {attrs}")
        
        for attr in attrs:
            try:
                value = getattr(result, attr)
                if callable(value):
                    print(f"   {attr}: <method>")
                else:
                    print(f"   {attr}: {value} (type: {type(value)})")
            except Exception as e:
                print(f"   {attr}: Error accessing - {e}")
        
        # Check metric state after evaluation
        print(f"\n📊 METRIC STATE AFTER EVALUATION:")
        metric_attrs = [attr for attr in dir(metric) if not attr.startswith('_') and not callable(getattr(metric, attr))]
        for attr in metric_attrs:
            try:
                value = getattr(metric, attr)
                print(f"   metric.{attr}: {value}")
            except Exception as e:
                print(f"   metric.{attr}: Error - {e}")
        
        return True, result, metric
        
    except Exception as e:
        print(f"❌ Simple evaluation failed: {e}")
        traceback.print_exc()
        return False, None, None

def test_assert_test_approach():
    """Test the assert_test approach"""
    print("\n" + "="*60)
    print("🔍 TESTING ASSERT_TEST APPROACH")
    print("="*60)
    
    try:
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric
        
        # Create test case
        test_case = LLMTestCase(
            input="What is Python?",
            actual_output="Python is a programming language"
        )
        
        # Create metric
        metric = AnswerRelevancyMetric(threshold=0.1)
        
        print("🔄 Running assert_test...")
        
        # This should either pass or raise an exception
        assert_test(test_case, [metric])
        
        print("✅ assert_test completed without exception")
        
        # Check metric state
        print(f"📊 Metric state after assert_test:")
        if hasattr(metric, 'score'):
            print(f"   score: {metric.score}")
        if hasattr(metric, 'success'):
            print(f"   success: {metric.success}")
        if hasattr(metric, 'reason'):
            print(f"   reason: {metric.reason}")
            
        return True, metric
        
    except Exception as e:
        print(f"❌ assert_test failed: {e}")
        traceback.print_exc()
        return False, None

def test_metric_measure_directly():
    """Test calling metric.measure() directly"""
    print("\n" + "="*60)
    print("🎯 TESTING DIRECT METRIC MEASUREMENT")
    print("="*60)
    
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric
        
        # Create test case
        test_case = LLMTestCase(
            input="What is AI?",
            actual_output="AI stands for Artificial Intelligence, which is a field of computer science"
        )
        
        # Create metric
        metric = AnswerRelevancyMetric(threshold=0.1)
        
        print("🔄 Calling metric.measure() directly...")
        
        # Try to call measure directly
        if hasattr(metric, 'measure'):
            result = metric.measure(test_case)
            print(f"✅ Direct measure() call completed")
            print(f"   Result: {result}")
        elif hasattr(metric, 'evaluate'):
            result = metric.evaluate(test_case)
            print(f"✅ Direct evaluate() call completed")
            print(f"   Result: {result}")
        else:
            print("⚠️ Metric has no measure() or evaluate() method")
            return False, None
        
        # Check all metric attributes after measurement
        print(f"📊 All metric attributes after measurement:")
        all_attrs = dir(metric)
        for attr in sorted(all_attrs):
            if not attr.startswith('_'):
                try:
                    value = getattr(metric, attr)
                    if not callable(value):
                        print(f"   {attr}: {value} (type: {type(value)})")
                except:
                    pass
        
        return True, metric
        
    except Exception as e:
        print(f"❌ Direct measurement failed: {e}")
        traceback.print_exc()
        return False, None

def run_comprehensive_diagnostic():
    """Run all diagnostic tests"""
    print("🚀 STARTING COMPREHENSIVE DEEPEVAL DIAGNOSTIC")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    
    # Load environment
    load_env()
    
    # Test 1: OpenAI API
    openai_works = test_openai_api()
    
    # Test 2: DeepEval imports
    deepeval_works = test_deepeval_imports()
    
    if not deepeval_works:
        print("\n❌ Cannot proceed - DeepEval imports failed")
        return
    
    if not openai_works:
        print("\n⚠️ OpenAI API failed - this is likely the root cause!")
        print("   - Check your API key")
        print("   - Verify you have credits")
        print("   - Check network connectivity")
    
    # Test 3: Simple DeepEval evaluation
    eval_works, result, metric = test_simple_deepeval()
    
    # Test 4: assert_test approach
    assert_works, assert_metric = test_assert_test_approach()
    
    # Test 5: Direct metric measurement
    direct_works, direct_metric = test_metric_measure_directly()
    
    # Summary
    print("\n" + "="*80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"OpenAI API:              {'✅ PASS' if openai_works else '❌ FAIL'}")
    print(f"DeepEval Imports:        {'✅ PASS' if deepeval_works else '❌ FAIL'}")
    print(f"Simple Evaluation:       {'✅ PASS' if eval_works else '❌ FAIL'}")
    print(f"assert_test Approach:    {'✅ PASS' if assert_works else '❌ FAIL'}")
    print(f"Direct Measurement:      {'✅ PASS' if direct_works else '❌ FAIL'}")
    
    # Root cause analysis
    print(f"\n🔍 ROOT CAUSE ANALYSIS:")
    if not openai_works:
        print("❌ PRIMARY ISSUE: OpenAI API is not working")
        print("   This is almost certainly why all scores are 0.0")
        print("   DeepEval cannot evaluate without a working LLM API")
    elif not eval_works and not assert_works and not direct_works:
        print("❌ PRIMARY ISSUE: DeepEval evaluation is completely broken")
        print("   This suggests a version compatibility or setup issue")
    elif eval_works or assert_works or direct_works:
        print("✅ DeepEval CAN work - the issue is in your test code")
        print("   Focus on fixing the score extraction logic")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if not openai_works:
        print("1. Fix OpenAI API access first")
        print("2. Verify API key in .env file")
        print("3. Check OpenAI account has credits")
        print("4. Test with: openai api completions.create -e davinci")
    else:
        print("1. Update DeepEval to latest version: pip install -U deepeval")
        print("2. Use working evaluation approach in your tests")
        print("3. Focus on score extraction from working method")

if __name__ == "__main__":
    run_comprehensive_diagnostic()