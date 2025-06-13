#!/usr/bin/env python3
"""
Demo script để test nhanh chức năng xử lý PDF với WDMPDFParser
"""

import os
import sys
from file_loader import PDFLoader


def quick_test(pdf_path: str):
    """Demo nhanh xử lý PDF"""
    print(f"🔍 Testing PDF: {pdf_path}")
    
    # Basic mode (không có credentials)
    print("\n📋 BASIC MODE (without credentials):")
    loader_basic = PDFLoader(debug=True)
    docs_basic = loader_basic.load(path_string=pdf_path)
    
    text_basic = [d for d in docs_basic if d.metadata.get("type") == "text"]
    table_basic = [d for d in docs_basic if d.metadata.get("type") == "table"]
    
    print(f"  ✅ Text chunks: {len(text_basic)}")
    print(f"  ✅ Tables: {len(table_basic)}")
    
    # Advanced mode (nếu có credentials)
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.exists(cred_path):
        print(f"\n🚀 ADVANCED MODE (with credentials):")
        print(f"   Using: {cred_path}")
        
        loader_advanced = PDFLoader(credential_path=cred_path, debug=True)
        docs_advanced = loader_advanced.load(path_string=pdf_path)
        
        text_advanced = [d for d in docs_advanced if d.metadata.get("type") == "text"]
        table_advanced = [d for d in docs_advanced if d.metadata.get("type") == "table"]
        
        print(f"  ✅ Text chunks: {len(text_advanced)}")
        print(f"  ✅ Tables (merged): {len(table_advanced)}")
        
        # So sánh
        print(f"\n📊 COMPARISON:")
        print(f"  Basic tables: {len(table_basic)}")
        print(f"  Advanced tables: {len(table_advanced)}")
        print(f"  Improvement: {len(table_basic) - len(table_advanced)} tables merged")
    else:
        print(f"\n⚠️ No credentials found - skipping advanced mode")
    
    print(f"\n✅ Demo completed!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demo_pdf_processing.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    quick_test(pdf_path) 