# 🧾 BÁO CÁO BÀI TẬP VỀ NHÀ -- XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)

## I. Giới thiệu chung

Báo cáo này trình bày kết quả hai bài thực hành cơ bản trong lĩnh vực
**Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing -- NLP):**

-   **Lab 1:** Cài đặt hai bộ tách từ (*Tokenizer*) là `SimpleTokenizer`
    và `RegexTokenizer`.
-   **Lab 2:** Sử dụng các tokenizer này để xây dựng bộ **Count
    Vectorizer**, nhằm biểu diễn văn bản thành dạng số (document-term
    matrix).

**Mục tiêu:**

-   Hiểu cơ chế tách từ trong văn bản (*Tokenization*).
-   Hiểu cách biểu diễn văn bản thành vector để phục vụ các mô hình học
    máy.
-   So sánh kết quả giữa các phương pháp tokenization khác nhau.

------------------------------------------------------------------------

## II. Các bước triển khai

### 1️⃣ Lab 1 -- Tokenization

#### a. SimpleTokenizer

**Cơ chế:**

-   Chuyển toàn bộ chuỗi về chữ thường (`lower()`).
-   Duyệt qua từng ký tự trong `string.punctuation`, chèn thêm khoảng
    trắng hai bên để tách riêng dấu câu.
-   Cuối cùng, tách thành danh sách token bằng `split()`.

**Mục tiêu:** Đơn giản, dễ hiểu, phù hợp với văn bản không quá phức tạp.

**Hạn chế:** Không xử lý tốt các mẫu đặc biệt (như từ ghép, dấu nháy
hoặc từ viết tắt).

------------------------------------------------------------------------

#### b. RegexTokenizer

**Cơ chế:**

-   Dùng biểu thức chính quy: `\w+|[^\w\s]` để nhận diện "chuỗi ký tự
    chữ" hoặc "ký tự đặc biệt không phải khoảng trắng".
-   Token hóa chính xác hơn khi gặp nhiều loại dấu câu hoặc ký tự không
    phải chữ.

**Ưu điểm:** Giữ nguyên dấu câu riêng biệt và xử lý nhất quán hơn
`SimpleTokenizer`.

------------------------------------------------------------------------

### 2️⃣ Lab 2 -- Count Vectorization

#### a. Cấu trúc

`CountVectorizer` nhận một Tokenizer làm tham số.\
Gồm hai phương thức chính:

-   **fit(corpus):** tạo từ điển (vocabulary) ánh xạ token → chỉ số.
-   **transform(docs):** chuyển từng văn bản thành vector đếm tần suất
    token.

#### b. Ví dụ hoạt động

-   **fit()**
    -   Token hóa toàn bộ corpus.
    -   Thu thập tất cả token duy nhất → sắp xếp → gán chỉ số.
-   **transform()**
    -   Với mỗi văn bản, khởi tạo vector độ dài = kích thước từ điển.
    -   Mỗi lần gặp token, tăng giá trị tương ứng lên 1.

**Kết quả:** ma trận document-term.

------------------------------------------------------------------------

## III. Cách chạy code và ghi log

**Cấu trúc thư mục:**

    lab1/
      ├── src/preprocessing/simple_tokenizer.py
      ├── src/preprocessing/regex_tokenizer.py
    lab2/
      ├── src/preprocessing/count_vectorizer.py
    main.py

**Chạy chương trình:**

``` bash
python main.py
```

**Kết quả được in ra log như sau:**

------------------------------------------------------------------------

## IV. Kết quả thực nghiệm và phân tích

### 1️⃣ Kết quả Lab 1 -- Tokenization

**Input:**\
`Hello, world! This is a test.`

  -----------------------------------------------------------------------
  Tokenizer                                Output
  ---------------------------------------- ------------------------------
  SimpleTokenizer                          \['hello', ',', 'world', '!',
                                           'this', 'is', 'a', 'test',
                                           '.'\]

  RegexTokenizer                           \['hello', ',', 'world', '!',
                                           'this', 'is', 'a', 'test',
                                           '.'\]
  -----------------------------------------------------------------------

→ *Nhận xét:* Hai phương pháp cho kết quả giống nhau trên câu cơ bản.

------------------------------------------------------------------------

**Input:**\
`NLP is fascinating... isn't it?`

  -----------------------------------------------------------------------
  Tokenizer                                Output
  ---------------------------------------- ------------------------------
  SimpleTokenizer                          \['nlp', 'is', 'fascinating',
                                           '.', '.', '.', 'isn', "'",
                                           't', 'it', '?'\]

  RegexTokenizer                           \['nlp', 'is', 'fascinating',
                                           '.', '.', '.', 'isn', "'",
                                           't', 'it', '?'\]
  -----------------------------------------------------------------------

→ *Nhận xét:* Cả hai đều nhận diện dấu chấm liên tiếp và dấu nháy đơn
`'` tách rời.\
Đây là nhược điểm của cách tokenization đơn giản -- không gộp các token
thành "isn't".

------------------------------------------------------------------------

**Input:**\
`Let's see how it handles 123 numbers and punctuation!`

  -----------------------------------------------------------------------
  Tokenizer                                Output
  ---------------------------------------- ------------------------------
  SimpleTokenizer                          \['let', "'", 's', 'see',
                                           'how', 'it', 'handles', '123',
                                           'numbers', 'and',
                                           'punctuation', '!'\]

  RegexTokenizer                           \['let', "'", 's', 'see',
                                           'how', 'it', 'handles', '123',
                                           'numbers', 'and',
                                           'punctuation', '!'\]
  -----------------------------------------------------------------------

→ *Nhận xét:* Cả hai nhận diện tốt số học (123) và dấu câu (!).\
RegexTokenizer thể hiện tính ổn định cao hơn khi xử lý các pattern đặc
biệt.

------------------------------------------------------------------------

**Trên tập UD English-EWT:**\
Ví dụ đoạn đầu:

> From the AP comes this story : President Bush on Tuesday nominated two
> individuals ...

Kết quả:

    SimpleTokenizer Output (first 20): ['from', 'the', 'ap', 'comes', 'this', 'story', ':', 'president', 'bush', ...]
    RegexTokenizer Output (first 20): ['from', 'the', 'ap', 'comes', 'this', 'story', ':', 'president', 'bush', ...]

→ *Nhận xét:* Hai tokenizer hoạt động tương đương trên văn bản chuẩn hóa
tin tức (ít ký tự đặc biệt).

------------------------------------------------------------------------

### 2️⃣ Kết quả Lab 2 -- Count Vectorization

#### Task 1: Toy Corpus

**Vocabulary:**

    {'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}

→ *Nhận xét:*

-   Từ điển gồm 10 token duy nhất, sắp xếp theo thứ tự alphabet.
-   Các vector thể hiện số lần xuất hiện của từng token trong từng văn
    bản.

**Document-term matrix:**

    [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]

→ Dễ thấy từ "." xuất hiện ở tất cả câu, "nlp" chỉ ở một số câu.

------------------------------------------------------------------------

#### Task 2: UD English-EWT Dataset

**Vocabulary (first 20 items):**

    {'-': 0, '.': 1, '15': 2, ':': 3, 'a': 4, 'anderson': 5, 'ap': 6, 'area': 7, 'as': 8, 'associate': 9, ...}

**Document-term matrix (first 5 docs):**

Một số câu rỗng → vector toàn 0.\
Các câu có nội dung dài → vector thưa (nhiều 0, ít 1).

→ *Nhận xét:*\
`CountVectorizer` hoạt động đúng nguyên lý, sinh vector sparse cho mỗi
câu.\
Với dữ liệu thật, kích thước từ vựng lớn nên vector dài và thưa.\
Đây là đặc điểm phổ biến của **Bag-of-Words representation**.

------------------------------------------------------------------------

## V. Khó khăn gặp phải và cách giải quyết

  ------------------------------------------------------------------------
  Vấn đề           Nguyên nhân               Cách khắc phục
  ---------------- ------------------------- -----------------------------
  Regex không bắt  Thiếu nhóm phủ định       Sử dụng pattern `\w+|[^\w\s]`
  đúng ký tự đặc   `\s`{=tex}                
  biệt                                       

  Vocabulary bị    `set` trong Python không  Thêm bước `sorted()` trước
  sắp xếp lộn xộn  có thứ tự                 khi `enumerate`
  giữa các lần                               
  chạy                                       

  Một số câu trong Dữ liệu UD chứa dòng      Lọc bỏ hoặc bỏ qua các
  dataset rỗng →   trống                     document trống khi tính
  vector toàn 0                              vector
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## VI. Nguồn tham khảo

-   Python documentation -- `re` module\
-   scikit-learn -- `CountVectorizer` API\
-   Universal Dependencies English-EWT Dataset

------------------------------------------------------------------------

## VII. Model / Prompt sử dụng

Không sử dụng mô hình NLP sẵn (như spaCy, NLTK, HuggingFace).\
Toàn bộ tokenization và vectorization được cài đặt thủ công bằng Python.

------------------------------------------------------------------------

## VIII. Kết luận

-   Hai bộ tokenizer hoạt động tương đương trên văn bản chuẩn, nhưng
    `RegexTokenizer` ổn định hơn khi gặp ký tự đặc biệt.\
-   `CountVectorizer` xây dựng thành công ma trận tần suất từ -- nền
    tảng cho TF-IDF và các mô hình học máy NLP sau này.

**Qua bài này, người học nắm vững quy trình cơ bản:**\
`Text → Tokens → Vocabulary → Count Vector.`
