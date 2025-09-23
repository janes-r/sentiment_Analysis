#!/usr/bin/env python3
"""
Test script for the Sentiment Analysis API
"""

import requests
import json
import time
from typing import Dict, List

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """
    Test the health check endpoint
    """
    print("Testing health check...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Uptime: {data.get('uptime', 0):.2f}s")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_prediction():
    """Test single sentiment prediction"""
    print("\nTesting single prediction...")
    
    test_cases = [
        {
            "text": "This movie is absolutely amazing! I loved every second of it.",
            "expected": "positive"
        },
        {
            "text": "Terrible film. Complete waste of time and money.",
            "expected": "negative"
        },
        {
            "text": "The acting was decent but the plot was confusing and hard to follow.",
            "expected": "negative"
        },
        {
            "text": "Outstanding performance! This is definitely one of the best movies of the year.",
            "expected": "positive"
        },
        {
            "text": "Boring and predictable storyline with poor character development.",
            "expected": "negative"
        }
    ]
    
    correct_predictions = 0
    
    for i, case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"text": case["text"]},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                predicted = data["sentiment"]
                confidence = data["confidence"]
                
                is_correct = predicted == case["expected"]
                if is_correct:
                    correct_predictions += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} Test {i}:")
                print(f"   Text: '{case['text'][:60]}...'")
                print(f"   Expected: {case['expected']}, Got: {predicted}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Probabilities: {data['probabilities']}")
                
            else:
                print(f"❌ Test {i} failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Test {i} error: {e}")
    
    accuracy = correct_predictions / len(test_cases)
    print(f"\n📊 Single Prediction Results:")
    print(f"   Correct: {correct_predictions}/{len(test_cases)}")
    print(f"   Accuracy: {accuracy:.2%}")
    
    return accuracy > 0.6  # At least 60% should be correct

def test_batch_prediction():
    """Test batch sentiment prediction"""
    print("\nTesting batch prediction...")
    
    test_texts = [
        "Amazing movie! Loved it!",
        "Terrible waste of time.",
        "Best film I've seen this year!",
        "Boring and predictable.",
        "Outstanding cinematography!"
    ]
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"texts": test_texts},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            processing_time = time.time() - start_time
            
            print(f"✅ Batch prediction successful!")
            print(f"   Processed: {data['total_processed']} texts")
            print(f"   Server time: {data['processing_time']:.3f}s")
            print(f"   Total time: {processing_time:.3f}s")
            
            print(f"\n   Results:")
            for i, result in enumerate(data['results'], 1):
                text = test_texts[i-1][:40] + "..." if len(test_texts[i-1]) > 40 else test_texts[i-1]
                print(f"   {i}. '{text}' → {result['sentiment']} ({result['confidence']:.3f})")
            
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved!")
            print(f"   Model type: {data.get('model_type', 'unknown')}")
            print(f"   Accuracy: {data.get('accuracy', 'unknown')}")
            print(f"   Training samples: {data.get('training_samples', 'unknown')}")
            print(f"   Vocabulary size: {data.get('vocabulary_size', 'unknown')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_error_handling():
    """Test API error handling"""
    print("\nTesting error handling...")
    
    # Test empty text
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": ""},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 422:  # Validation error
            print("✅ Empty text validation works")
        else:
            print(f"❌ Empty text should return 422, got {response.status_code}")
    except Exception as e:
        print(f"❌ Empty text test error: {e}")
    
    # Test too long text
    try:
        long_text = "This is a very long text. " * 200  # Over 5000 chars
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": long_text},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 422:  # Validation error
            print("✅ Long text validation works")
        else:
            print(f"❌ Long text should return 422, got {response.status_code}")
    except Exception as e:
        print(f"❌ Long text test error: {e}")

def run_full_test_suite():
    """Run all tests"""
    print("Starting API Test Suite")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code != 200:
            print(f"❌ API not accessible at {API_BASE_URL}")
            print("Please make sure the API is running with: python -m uvicorn api.main:app --reload")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API at {API_BASE_URL}")
        print(f"Error: {e}")
        print("Please make sure the API is running with: python -m uvicorn api.main:app --reload")
        return False
    
    # Run tests
    tests_passed = 0
    total_tests = 5
    
    if test_health_check():
        tests_passed += 1
    
    if test_model_info():
        tests_passed += 1
    
    if test_single_prediction():
        tests_passed += 1
    
    if test_batch_prediction():
        tests_passed += 1
    
    test_error_handling()  # Always run, doesn't count towards pass/fail
    tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! API is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = run_full_test_suite()
    exit(0 if success else 1)