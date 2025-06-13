# Table Context Enhancement Feature

## Tổng quan / Overview

Tính năng này giải quyết vấn đề khi các bảng lớn được chia thành các chunk nhỏ trong quá trình xử lý RAG. Khi người dùng hỏi về thông tin cần toàn bộ bảng, nhưng chỉ có một chunk nhỏ được truy xuất, hệ thống sẽ tự động lấy toàn bộ bảng để cung cấp context đầy đủ cho LLM.

This feature solves the problem when large tables are split into small chunks during RAG processing. When users ask for information requiring the full table context, but only a small chunk is retrieved, the system automatically retrieves the entire table to provide complete context for the LLM.

## Cách hoạt động / How it works

### 1. Phát hiện Table Chunks / Table Chunk Detection

Hệ thống sử dụng các từ khóa để phát hiện khi một chunk được truy xuất có chứa thông tin bảng:

The system uses keywords to detect when a retrieved chunk contains table information:

```python
TABLE_DETECTION_KEYWORDS = [
    "table content:", "number of rows:", "number of columns:", 
    "headers:", "this is a merged table", "this table has headers",
    "table from page", "merged table spanning"
]
```

### 2. Truy xuất Full Table Context / Full Table Context Retrieval

Khi phát hiện table chunk, hệ thống sẽ:

When a table chunk is detected, the system will:

1. Xác định source document / Identify the source document
2. Truy xuất tất cả table documents từ cùng source / Retrieve all table documents from the same source
3. Sắp xếp theo thứ tự trang / Sort by page order
4. Kết hợp thành context đầy đủ / Combine into complete context

### 3. Enhanced Generation

Context được cải thiện sẽ được sử dụng trong generation stage thay vì chunk nhỏ ban đầu.

The enhanced context is used in the generation stage instead of the original small chunk.

## Cấu hình / Configuration

### Trong file `setting.py`:

```python
# Bật/tắt tính năng / Enable/disable feature
ENABLE_TABLE_CONTEXT_ENHANCEMENT = True

# Giới hạn kích thước context (ký tự) / Context size limit (characters)
MAX_TABLE_CONTEXT_SIZE = 10000

# Từ khóa phát hiện bảng / Table detection keywords
TABLE_DETECTION_KEYWORDS = [
    "table content:", "number of rows:", "number of columns:", 
    "headers:", "this is a merged table", "this table has headers",
    "table from page", "merged table spanning"
]
```

## Sử dụng / Usage

### Tự động / Automatic

Tính năng hoạt động tự động trong quy trình RAG. Không cần thay đổi code sử dụng.

The feature works automatically in the RAG workflow. No changes needed in usage code.

```python
# Khởi tạo workflow như bình thường
workflow = WorkFlow()

# Tính năng sẽ tự động hoạt động khi có table chunks được truy xuất
# Feature will automatically activate when table chunks are retrieved
```

### Thủ công / Manual

Bạn có thể sử dụng các method mới trong VectorStore:

You can use the new methods in VectorStore:

```python
# Lấy tất cả tables từ một source
tables = vector_store.get_tables_from_source("document.pdf")

# Lấy documents theo metadata
docs = vector_store.get_document_by_metadata({"type": "table", "page": 1})
```

## API Methods

### VectorStore

#### `get_tables_from_source(source_name: str) -> List[Document]`

Truy xuất tất cả table documents từ một source cụ thể.

Retrieve all table documents from a specific source.

**Parameters:**
- `source_name`: Tên file source

**Returns:**
- List of Document objects chứa tables, được sắp xếp theo trang

#### `get_document_by_metadata(metadata_filters: dict) -> List[Document]`

Truy xuất documents dựa trên metadata filters.

Retrieve documents based on metadata filters.

**Parameters:**
- `metadata_filters`: Dictionary của key-value pairs để filter

**Returns:**
- List of Document objects matching the filters

### WorkFlow

#### `_get_full_table_context(retrieved_docs: str) -> str`

Phát hiện table chunks và mở rộng thành full table context.

Detect table chunks and expand to full table context.

**Parameters:**
- `retrieved_docs`: Content được truy xuất từ vector store

**Returns:**
- Enhanced context với full table information

## Ví dụ / Examples

### Kịch bản 1: Table bị chia thành chunks

```python
# Trước khi có tính năng này / Before this feature:
# Chỉ truy xuất được một phần bảng
retrieved_chunk = """
Name: John Doe, Age: 30
Name: Jane Smith, Age: 28
"""

# Sau khi có tính năng này / After this feature:
# Tự động truy xuất toàn bộ bảng
full_context = """
=== FULL TABLE CONTEXT ===
Table from page 1:
Name: John Doe, Age: 30, Department: Engineering, Salary: $75000
Name: Jane Smith, Age: 28, Department: Marketing, Salary: $65000
Name: Bob Johnson, Age: 35, Department: Engineering, Salary: $85000
...
"""
```

### Kịch bản 2: Câu hỏi về toàn bộ bảng

```python
# Câu hỏi người dùng
user_question = "Tổng lương của tất cả nhân viên là bao nhiều?"

# Trước: Chỉ có thông tin một phần → Không thể trả lời chính xác
# Sau: Có toàn bộ bảng → Có thể tính toán chính xác
```

## Testing

Chạy test script để kiểm tra tính năng:

Run the test script to check the feature:

```bash
python test_table_enhancement.py
```

Test script sẽ:
- Tạo sample table documents
- Mô phỏng việc truy xuất table chunks
- Kiểm tra tính năng enhancement
- Hiển thị kết quả so sánh

## Troubleshooting

### Vấn đề phổ biến / Common Issues

1. **Tính năng không hoạt động**
   - Kiểm tra `ENABLE_TABLE_CONTEXT_ENHANCEMENT = True`
   - Kiểm tra table chunks có chứa keywords detection không

2. **Context quá lớn**
   - Điều chỉnh `MAX_TABLE_CONTEXT_SIZE`
   - Hệ thống sẽ tự động truncate nếu vượt quá giới hạn

3. **Không tìm thấy full table**
   - Kiểm tra metadata `source` và `type` có đúng không
   - Kiểm tra table documents có được lưu trong vectorstore không

### Logging

Hệ thống ghi log chi tiết về quá trình enhancement:

```
##Enhanced Table Context: Found and expanded 2 table sources
##Generate Task: Enhanced context with full table information
```

## Hiệu suất / Performance

- Tính năng chỉ kích hoạt khi phát hiện table chunks
- Sử dụng cache để tránh truy xuất lặp lại
- Có giới hạn kích thước để tránh quá tải memory

## Tương lai / Future Enhancements

- Hỗ trợ cross-document table linking
- Intelligent table merging
- Table relationship detection
- Advanced table context optimization 