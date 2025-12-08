"""
Diagnostic Script for Image Matching Issue

Run this script to test the image filename pattern matching
and see what's happening with your images.
"""

import re
from pathlib import Path

def extract_page_number_old(filename: str) -> int:
    """Old version - single pattern"""
    match = re.search(r'-page-(\d+)\.jpg$', str(filename))
    return int(match.group(1)) if match else float('inf')

def extract_page_number_new(filename: str) -> int:
    """New version - multiple patterns"""
    patterns = [
        r'-page-(\d+)\.jpg$',      # document-page-5.jpg
        r'page-(\d+)\.jpg$',       # page-5.jpg
        r'page_(\d+)\.jpg$',       # page_5.jpg
        r'_(\d+)\.jpg$',           # document_5.jpg
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(filename), re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    print(f"  WARNING: Could not extract page number from: {filename}")
    return float('inf')

def test_pattern_matching():
    """Test various filename patterns"""
    
    test_filenames = [
        "document-page-1.jpg",
        "document-page-5.jpg",
        "page-1.jpg",
        "page-10.jpg",
        "Page-1.jpg",  # uppercase
        "rjf1q25-page-1.jpg",
        "rjf1q25-page-25.jpg",
        "page_1.jpg",
        "document_5.jpg",
        "some_weird_name.jpg",  # should fail
    ]
    
    print("=" * 70)
    print("PATTERN MATCHING TEST")
    print("=" * 70)
    print(f"{'Filename':<30} {'Old Method':<15} {'New Method':<15}")
    print("-" * 70)
    
    for filename in test_filenames:
        old_result = extract_page_number_old(filename)
        new_result = extract_page_number_new(filename)
        
        old_str = str(old_result) if old_result != float('inf') else "FAILED"
        new_str = str(new_result) if new_result != float('inf') else "FAILED"
        
        print(f"{filename:<30} {old_str:<15} {new_str:<15}")
    
    print()

def check_actual_images(images_dir: str):
    """Check actual images in your directory"""
    
    print("=" * 70)
    print("ACTUAL IMAGES IN YOUR DIRECTORY")
    print("=" * 70)
    
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"ERROR: Directory doesn't exist: {images_dir}")
        print("Please update the path in this script.")
        return
    
    image_files = sorted([
        f for f in images_path.iterdir() 
        if f.is_file() and f.suffix.lower() == '.jpg'
    ])
    
    if not image_files:
        print(f"No .jpg images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images\n")
    print(f"{'#':<5} {'Filename':<40} {'Old':<10} {'New':<10}")
    print("-" * 70)
    
    for i, img in enumerate(image_files):
        old_page = extract_page_number_old(img.name)
        new_page = extract_page_number_new(img.name)
        
        old_str = str(old_page) if old_page != float('inf') else "FAIL"
        new_str = str(new_page) if new_page != float('inf') else "FAIL"
        
        print(f"{i:<5} {img.name:<40} {old_str:<10} {new_str:<10}")
    
    print()
    
    # Check for sorting issues
    print("=" * 70)
    print("SORTING CHECK")
    print("=" * 70)
    
    # Sort using old method
    sorted_old = sorted(image_files, key=lambda f: extract_page_number_old(f.name))
    # Sort using new method
    sorted_new = sorted(image_files, key=lambda f: extract_page_number_new(f.name))
    
    print("Order using OLD pattern matching:")
    for i, img in enumerate(sorted_old[:10]):  # Show first 10
        page = extract_page_number_old(img.name)
        page_str = str(page) if page != float('inf') else "UNKNOWN"
        print(f"  {i+1}. {img.name} -> page {page_str}")
    
    print("\nOrder using NEW pattern matching:")
    for i, img in enumerate(sorted_new[:10]):  # Show first 10
        page = extract_page_number_new(img.name)
        page_str = str(page) if page != float('inf') else "UNKNOWN"
        print(f"  {i+1}. {img.name} -> page {page_str}")
    
    print()

if __name__ == "__main__":

    # Test the pattern matching
    test_pattern_matching()
    
    # Check your actual images
    # ⚠️ UPDATE THIS PATH to your actual images directory
    # images_directory = "data/images"
    images_directory = "E:\Python\Repos\portfolio\Financial Chat Assistant\data\images"
    
    # Alternative common paths:
    # images_directory = "data\\images"  # Windows
    # images_directory = "/path/to/your/project/data/images"  # Absolute path
    
    print(f"\nChecking images in: {images_directory}")
    print("(Update the path in this script if needed)\n")
    
    check_actual_images(images_directory)
    
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. Check if the 'New Method' correctly extracts page numbers from your images
2. If page numbers are extracted correctly, replace the extract_page_number() 
   method in indexing.py with the new version
3. If page numbers still fail, the images may have unexpected naming patterns
   - Check the actual filenames above
   - Add the pattern to the 'patterns' list in extract_page_number_new()
4. Apply the other fixes from image_matching_fix.md
    """)
