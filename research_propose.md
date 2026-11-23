# **ĐỀ XUẤT NGHIÊN CỨU KHOA HỌC (RESEARCH PROPOSAL)**

**Tiêu đề Dự kiến (Tiếng Anh):** *Unmasking the Hivemind: Disentangling Post-Training Knowledge Collapse and Generation Order Degeneration in Discrete Diffusion Language Models***Tiêu đề Dự kiến (Tiếng Việt):** *Giải phẫu Bầy đàn Nhân tạo: Phân tách Sụp đổ Tri thức Hậu huấn luyện và Thoái hóa Thứ tự Giải mã trong Các Mô hình Ngôn ngữ Khuếch tán Rời rạc***Hội nghị & Phân ban Mục tiêu:** ACL 2027 — Main Conference, **Special Theme Track**: *Homogenization and Knowledge Collapse in LLMs***Chu kỳ Nộp bài:** ACL Rolling Review (ARR) Tháng 01/2027 (Hạn nộp: **04/01/2027 AoE**) **Hạ tầng Thực nghiệm:** 1× GPU NVIDIA T4 (16 GB VRAM, kiến trúc Turing sm\_75) — Thiết kế chẩn đoán quỹ đạo suy luận không cần huấn luyện lại (*Training-Free Trajectory Diagnostics*)

## **1\. Tóm tắt Đề xuất (Executive Summary)**

Hiện tượng **đồng nhất hóa đầu ra (homogenization)** và **sụp đổ tri thức đuôi dài (knowledge collapse)** sau giai đoạn hậu huấn luyện (*post-training alignment*) đang trở thành một trong những thách thức trung tâm của xử lý ngôn ngữ tự nhiên hiện đại. Tuy nhiên, gần như toàn bộ tri thức lý thuyết và thực nghiệm hiện có đều mặc định kiến trúc **tự hồi quy (Autoregressive — AR)**, nơi văn bản được sinh tuần tự từ trái sang phải. Sự trỗi dậy của thế hệ **Mô hình Ngôn ngữ Khuếch tán Rời rạc (Discrete Diffusion Language Models — dLLMs)** quy mô 7B–8B tham số như **Dream-7B** và **LLaDA-8B** đặt ra một câu hỏi nền tảng chưa có lời giải: *Cơ chế sinh văn bản phi tuyến tính thông qua giải mặt nạ lặp (iterative unmasking) giúp dLLM kháng cự tốt hơn trước sụp đổ tri thức hậu huấn luyện, hay lại tạo ra những chế độ thất bại (failure modes) hoàn toàn mới?*  
Đề xuất này xây dựng một **khung khảo sát chẩn đoán nhân quả có kiểm soát** nhằm nhận diện, định lượng và giải phẫu cơ chế nội tại của bốn chế độ thất bại đặc thù trên dLLM khi chuyển từ giai đoạn tiền huấn luyện (Base) sang căn chỉnh chỉ dẫn (Instruct). Điểm đột phá về phương pháp luận nằm ở việc tận dụng cặp mô hình song sinh **Dream-7B / Qwen2.5-7B** (vốn dùng chung trọng số khởi tạo) kết hợp với **giải phẫu quỹ đạo theo trục thời gian khử nhiễu $t=1\rightarrow T$** trực tiếp trên phần cứng phổ thông (**1× GPU T4 16 GB VRAM**). Từ các bằng chứng thực nghiệm thu được, nghiên cứu sẽ chốt một bài toán trung tâm sắc bén và đề xuất cơ chế điều tiết giải mặt nạ động theo pha thời gian tại lúc suy luận (*Phase-Aware Inference-Time Mitigation*), đáp ứng trọn vẹn tiêu chuẩn đóng góp kép (chẩn đoán cơ chế \+ giải pháp kỹ thuật) của **ACL 2027 Main Conference**.

## **2\. Bối cảnh và Động lực Nghiên cứu (Background & Motivation)**

### **2.1. Hiện tượng Đồng nhất hóa và Sụp đổ Tri thức Hậu huấn luyện**

Các nghiên cứu quy mô lớn giai đoạn 2025–2026 đã chứng minh rằng quá trình căn chỉnh (SFT, DPO, RLVR) không chỉ cải thiện khả năng tuân thủ chỉ dẫn mà còn đẩy các mô hình ngôn ngữ vào trạng thái **"đơn canh sinh tạo" (generative monoculture)**:

* **Hội tụ nội mô hình và liên mô hình (*Artificial Hivemind*):** Ngay cả dưới chế độ lấy mẫu ngẫu nhiên đa dạng, các LLM độc lập vẫn lặp lại cùng một số ít ẩn dụ, khuôn mẫu lập luận và cấu trúc diễn ngôn, trong khi đa dạng nhận thức ở cấp độ mệnh đề sự kiện (epistemic claims) thấp hơn đáng kể so với nguồn tri thức web tự nhiên.  
* **Cơ chế làm sắc phân phối theo hàm mũ (*Typicality Bias*):** Do thiên lệch điển hình của người gán nhãn ưu tiên (\$\\alpha \\log \\pi\_{\\text{ref}}(y \\mid x)\$), hàm tối ưu chính sách có phạt KL đẩy phân phối hậu huấn luyện về dạng \$\\pi^\*(y \\mid x) \\propto \\pi\_{\\text{ref}}(y \\mid x)^{1 \+ \\alpha/\\beta} \\exp(r\_{\\text{true}}(x,y)/\\beta)\$. Với hệ số làm sắc điển hình $\gamma =1+\alpha /\beta \approx 6.7$, xác suất của các đáp án sự kiện đuôi dài hợp lệ bị nén giảm hàng tỷ lần so với các cụm từ khuôn mẫu phổ biến, gây ra hiện tượng **kìm nén tri thức (suppression)** thay vì tẩy xóa hoàn toàn khỏi trọng số.  
* **Nghịch lý đo lường Đa chiều (Meaning vs. Form):** Các kiểm toán cuối năm 2026 chỉ ra rằng chỉ dựa vào độ tương đồng embedding toàn văn sẽ che khuất sự phân kỳ giữa *hội tụ hình thức trình bày* (mẫu câu mở/đóng, danh sách) với *sụp đổ nội dung tri thức thực chất*.

### **2.2. Khoảng trống Y văn: Discrete Diffusion Language Models (dLLMs)**

Khác với mô hình tự hồi quy (AR-LM) bị khóa chặt vào thứ tự nhân quả trái-sang-phải, mô hình ngôn ngữ khuếch tán rời rạc (DLM / dLLM) khởi đầu từ chuỗi toàn ký tự \[MASK\] và lặp qua $T$ bước khử nhiễu song song: ở mỗi bước $t$, mô hình dự đoán phân phối từ vựng trên toàn bộ chuỗi với ngữ cảnh hai chiều, giữ lại một tập con token đạt tiêu chí tự tin và đánh mặt nạ lại phần còn lại (*remasking*).  
Sự khác biệt kiến trúc này tạo ra **ba khoảng trống nghiên cứu lớn** chưa được khai phá tính đến cuối năm 2026:

> 1. **Thiếu vắng thực chứng về Knowledge Collapse trên dLLM:** Toàn bộ các nghiên cứu về *Artificial Hivemind* và *Typicality Sharpening* đều thực hiện trên họ AR (Llama, Qwen, Gemma, GPT). Chưa có nghiên cứu nào kiểm định xem dLLM chịu tác động của hậu huấn luyện lên tri thức đa đáp án và thực thể đuôi dài như thế nào.  
> 2. **Chưa giải phẫu trục thời gian khử nhiễu $t$:** DLM sở hữu một chiều dữ liệu nội tại mà AR không có: **quỹ đạo mở khóa token qua các bước thời gian $t\in [1,T]$**. Sự sụp đổ tri thức ở dLLM có thể diễn ra theo quy luật hoàn toàn khác — chẳng hạn đóng cứng khung cấu trúc câu ở pha đầu và bóp nghẹt thực thể hiếm ở pha cuối.  
> 3. **Định kiến về chi phí tính toán:** Các khảo sát *Model Collapse* truyền thống thường huấn luyện lại mô hình qua 5–8 thế hệ dữ liệu tổng hợp, đòi hỏi cụm đa GPU A100/H100 và dễ nhiễu bởi siêu tham số fine-tuning. Trên thực tế, việc đối chiếu trực tiếp cặp checkpoint Base $\rightarrow$ Instruct kết hợp trích xuất lịch sử quỹ đạo suy luận cho phép cô lập chính xác tác động của hậu huấn luyện ngay trên **1 GPU 16 GB VRAM**.

### **2.3. Tính Khả thi và Giá trị Chiến lược cho ACL 2027**

* **Lợi thế trực tiếp từ Special Theme ACL 2027:** Chủ đề *Homogenization and Knowledge Collapse in LLMs* ưu tiên các nghiên cứu phân tích cơ chế huấn luyện–căn chỉnh–giải mã gây ra đơn canh sinh tạo và đề xuất biện pháp giảm thiểu, với tỷ lệ chấp nhận lịch sử cao hơn mặt bằng toàn hội nghị từ 8–20 điểm phần trăm.  
* **Thiết kế đối chứng nhân quả sạch nhiễu (Twin-Model Control):** Mô hình **Dream-7B** được khởi tạo trực tiếp từ trọng số tiền huấn luyện của **Qwen2.5-7B** trước khi chuyển đổi sang mục tiêu khuếch tán. So sánh song song cặp Dream-7B (Base vs. Instruct) với Qwen2.5-7B (Base vs. Instruct) loại bỏ gần như hoàn toàn biến nhiễu do khác biệt dữ liệu tiền huấn luyện, cô lập thuần túy tác động của **cơ chế giải mã khuếch tán**.

## **3\. Câu hỏi Nghiên cứu Trọng tâm (Research Questions)**

Đề xuất này hướng tới trả lời ba câu hỏi nghiên cứu có thể kiểm chứng định lượng (falsifiable):

* **RQ1 (Hành vi Sụp đổ Tri thức):** Khi chuyển từ mô hình tiền huấn luyện (Base) sang mô hình căn chỉnh chỉ dẫn (Instruct), mức độ xói mòn độ phủ tri thức đa đáp án (*Recall@K*) và tri thức thực thể đuôi dài (*Tail Retention*) trên các dLLM (**Dream-7B, LLaDA-8B**) diễn ra mạnh hơn hay yếu hơn so với các mô hình tự hồi quy tương đương (**Qwen2.5-7B, Llama-3.1-8B**)?  
* **RQ2 (Cơ chế Quỹ đạo Nội tại):** Quá trình hậu huấn luyện biến dạng **động học giải mặt nạ theo thời gian ($t=1\rightarrow T$)** của dLLM như thế nào — cụ thể qua hiện tượng khóa khung cú pháp sớm, sụp đổ tính phi tuyến trong thứ tự mở khóa token, và hiệu ứng dây chuyền ảo giác hai chiều?  
* **RQ3 (Khả năng Phục hồi tại Lúc Suy luận):** Liệu việc điều tiết động chiến lược chọn vị trí và nhiệt độ giải mặt nạ theo từng pha thời gian khử nhiễu (*Phase-Aware Remasking*) có thể khôi phục độ phủ tri thức đuôi dài và đa dạng nhận thức mà không làm suy giảm độ chính xác sự kiện hay độ trôi chảy ngữ pháp?

## **4\. Phân loại và Định nghĩa Các Chế độ Thất bại Khả thi (Candidate Failure Modes)**

Nghiên cứu xác định **4 chế độ thất bại (Failure Modes — FM)** tiềm năng đặc thù cho kiến trúc khuếch tán rời rạc. Giai đoạn khảo sát thí điểm (Pilot Study) trên T4 16 GB sẽ kiểm định đồng thời cả 4 FM để chọn ra 1–2 FM có tín hiệu thống kê mạnh nhất làm trục xương sống cho bài báo.

### **FM-1: Early Structural Lock-in & Tail Knowledge Erasure (Khóa khung cấu trúc sớm & Xóa sổ tri thức đuôi dài)**

* **Cơ chế giả thuyết:** Trong dLLM, ở pha đầu của quá trình khử nhiễu ($t\in [1,0.25T]$), thuật toán chọn token theo độ tự tin ưu tiên mở khóa các cụm từ khuôn mẫu (*"The capital of..."*, *"Here are some..."*) vì xác suất biên của chúng cao nhất. Sau hậu huấn luyện (Instruct), hiện tượng làm sắc phân phối khiến các cụm từ khuôn mẫu này bị đóng cứng ngay ở những bước đầu tiên, thu hẹp mạnh không gian điều kiện hai chiều cho các thực thể hiếm ở pha tinh chỉnh sau ($t\rightarrow T$).  
* **Định nghĩa định lượng:**  
  * Tỷ lệ khóa khung sớm (**Early Boilerplate Lock-in Rate — EBLR**): Tỷ lệ token mở khóa trong 25% bước đầu tiên thuộc tập từ chức năng, dấu câu hoặc mẫu câu mở đầu cố định.

Tỷ số bảo toàn tri thức đuôi dài (**Tail-to-Head Retention Ratio — THR**):

$$THR=\frac{Ac{c}_{Tail}(Instruct)/Ac{c}_{Tail}(Base)}{Ac{c}_{Head}(Instruct)/Ac{c}_{Head}(Base)}$$

*   
* **Tiêu chí xác nhận:** Khi chuyển từ Base sang Instruct, EBLR tăng có ý nghĩa thống kê ($p<0.01$) ngay trong 16 bước đầu tiên, kéo theo Recall@16 trên CoverageQA và THR trên PopQA của dLLM giảm sâu hơn so với AR (Qwen2.5-7B), trong khi Precision@16 vẫn duy trì ở mức cao.

### **FM-2: Generation Order Collapse / The Flexibility Trap (Sụp đổ thứ tự giải mặt nạ phi tuyến)**

* **Cơ chế giả thuyết:** Lợi thế kiến trúc cốt lõi của dLLM là khả năng lập kế hoạch phi tuyến tính (mở khóa các thực thể mỏ neo ở giữa/cuối câu trước rồi mới điền từ nối xung quanh). Quá trình căn chỉnh chỉ dẫn (Instruct) triệt tiêu sự linh hoạt này, ép quỹ đạo mở khóa token co cụm về một thứ tự tuyến tính trái-sang-phải cứng nhắc qua mọi lần lấy mẫu, từ đó xóa bỏ đa dạng ý tưởng đầu ra.  
* **Định nghĩa định lượng:** Với mỗi lần chạy \$k \\in \\{1, \\dots, K\\}\$ và vị trí token \$j \\in \\{1, \\dots, L\\}\$, ký hiệu bước đầu tiên vị trí $j$ được giải mặt nạ là \$\\tau\_{k,j} \= \\min \\{t \\mid x^{(t)}\_{k,j} \\neq \\text{\[MASK\]}\\}\$.  
  * **Unmasking Order Entropy (${H}_{order}$, bits):** Entropy Shannon trung bình theo vị trí $j$ của phân phối \$\\{\\tau\_{1,j}, \\dots, \\tau\_{K,j}\\}\$ qua $K$ quỹ đạo độc lập. ${H}_{order}\rightarrow 0$ biểu thị sự đông cứng hoàn toàn về thứ tự sinh.  
  * **Left-to-Right Lock-in Correlation (${\rho }_{LTR}$):** Hệ số tương quan hạng Spearman giữa chỉ số vị trí $j$ và bước mở khóa ${\tau }_{k,j}$. ${\rho }_{LTR}\rightarrow +1.0$ chứng minh dLLM đã thoái hóa về hành vi tự hồi quy trái-sang-phải.  
* **Tiêu chí xác nhận:** Trên bản Instruct, ${H}_{order}$ giảm mạnh và ${\rho }_{LTR}$ dịch chuyển tiệm cận \+1.0 so với Base, đồng thời có tương quan thuận có ý nghĩa với mức sụt giảm đa dạng ngữ nghĩa **Vendi Score bậc $q=0.5$** ($V{S}_{0.5}$).

### **FM-3: Bidirectional Hallucination Cascade (Hiệu ứng dây chuyền ảo giác hai chiều)**

* **Cơ chế giả thuyết:** Vì lớp attention của dLLM là hai chiều trên toàn chuỗi, nếu mô hình chốt nhầm một thực thể cầu nối ở pha đầu (${t}_{unmask}\leq 0.25T$) — đặc biệt khi ép nén số bước khử nhiễu $T$ để tăng tốc suy luận — toàn bộ các bước khử nhiễu phía sau sẽ điều kiện hóa hai chiều trên thực thể sai này và tự hợp thức hóa nó thành một đoạn giải thích trôi chảy nhưng bịa đặt hoàn toàn.  
* **Định nghĩa định lượng:**  
  * **Tỷ lệ ảo giác tự tin (Hallucination Rate):** $1-Precision@K$ tính trên tập các phản hồi không từ chối trả lời.

**Độ trôi tự tin tự củng cố (Self-Confirmation Confidence Drift — $\Delta {C}_{wrong}$):**

$$\Delta {C}_{wrong}={P}^{(T)}({y}_{wrong})-{P}^{({t}_{unmask})}({y}_{wrong})$$

*   
  đo mức tăng xác suất của token thực thể sai từ lúc mới mở khóa (${t}_{unmask}$) tới bước cuối cùng ($t=T$).  
* **Tiêu chí xác nhận:** Khi nén số bước khử nhiễu $T\in \{128\rightarrow 64\rightarrow 32\rightarrow 16\}$, tỷ lệ ảo giác của dLLM tăng dốc hơn hẳn AR baseline ở mức ngân sách tính toán tương đương, và $\Delta {C}_{wrong}>0$ trên đa số mẫu ảo giác.

### **FM-4: Remasking Strategy Phase Collapse (Sụp đổ pha theo chiến lược giải mặt nạ tĩnh)**

* **Cơ chế giả thuyết:** Các thuật toán chọn vị trí giải mặt nạ hiện hành (maskgit\_plus theo Top-1 confidence, topk\_margin, entropy, origin ngẫu nhiên) áp dụng một quy tắc tĩnh xuyên suốt $t=1\rightarrow T$. Chiến lược tham lam theo độ tự tin gây đồng nhất hóa tri thức nặng nề ở pha đầu, trong khi chiến lược ngẫu nhiên làm phá vỡ tính nhất quán cú pháp ở pha cuối — tạo ra một đường biên đánh đổi Pareto cố hữu giữa *Đa dạng* và *Độ chính xác/Trôi chảy*.  
* **Định nghĩa định lượng:**  
  * **Algorithmic Collapse Gap (${\Delta }_{alg}$):** Độ sụt giảm Recall@16 và $V{S}_{0.5}$ giữa chiến lược ngẫu nhiên (origin) và chiến lược tham lam (maskgit\_plus) trên bản Instruct.  
  * **External Perplexity (PPL):** Độ hỗn loạn ngữ pháp đo độc lập bởi mô hình ngoài Qwen2.5-7B-Instruct.  
* **Tiêu chí xác nhận:** Không tồn tại chiến lược tĩnh đơn lẻ nào thống trị đồng thời cả hai trục; maskgit\_plus/entropy làm sụp đổ đa dạng tri thức trong khi origin làm tăng vọt External PPL, chứng minh tính cấp thiết của cơ chế điều tiết động theo pha thời gian.

## **5\. Thiết kế Thực nghiệm Chi tiết (Detailed Experimental Design)**

### **5.1. Kiến trúc Hệ quy chiếu Đối đầu và Chuẩn hóa Phần cứng T4 16 GB**

Để đảm bảo tính tái lập tuyệt đối và loại bỏ nhiễu phần cứng trên GPU **NVIDIA T4 (16 GB VRAM, kiến trúc Turing sm\_75)**, toàn bộ thí nghiệm tuân thủ 4 ràng buộc kỹ thuật:

> 1. **Lượng tử hóa đối xứng 4-bit NF4 (Twin Quantization):** Tất cả mô hình 7B–8B (cả dLLM lẫn AR baseline) đều được tải qua bitsandbytes với cấu hình load\_in\_4bit=True, bnb\_4bit\_quant\_type="nf4", bnb\_4bit\_use\_double\_quant=True. Việc lượng tử hóa đồng nhất cả hai họ mô hình loại trừ thiên lệch do sai số nén trọng số.  
> 2. **Định dạng tính toán torch.float16:** Bắt buộc dùng fp16 thay vì bfloat16 (vì T4 thiếu nhân Tensor Core bf16, chạy bf16 sẽ rơi về giả lập fp32 gây chậm 3–4× và tràn bộ nhớ).  
> 3. **Cơ chế Attention sdpa:** Sử dụng Scaled Dot-Product Attention chuẩn của PyTorch (attn\_implementation="sdpa") thay cho FlashAttention-2 (vốn yêu cầu kiến trúc Ampere sm\_80+).  
> 4. **Kiểm soát kích thước lô (batch\_size \= 4):** Giữ đỉnh VRAM ở mức \~11–12 GB ngay cả khi bật output\_history=True với 64 bước khử nhiễu.

| Nhóm Hệ quy chiếu | Mô hình dLLM (4-bit NF4 \+ fp16) | Mô hình AR Đối chứng (4-bit NF4 \+ fp16) | Đỉnh VRAM (Batch=4, 64 steps) | Vai trò Khoa học |
| :---- | :---- | :---- | :---- | :---- |
| **Cặp Chính (Cùng trọng số khởi tạo)** | Dream-org/Dream-v0-Base-7B Dream-org/Dream-v0-Instruct-7B | Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct | \~11,2 GB | Cô lập thuần túy cơ chế khuếch tán so với tự hồi quy trên cùng nền tri thức tiền huấn luyện. |
| **Cặp Kiểm chứng chéo (Huấn luyện từ đầu)** | GSAI-ML/LLaDA-8B-Base GSAI-ML/LLaDA-8B-Instruct | meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct | \~12,0 GB | Xác minh tính phổ quát của các FM trên dLLM được huấn luyện khuếch tán từ đầu (*pretrained from scratch*). |

### **5.2. Hệ thống Bộ Dữ liệu Phân tầng (405 Truy vấn Chuẩn hóa)**

Thay vì chạy tràn lan trên các benchmark QA đơn đáp án truyền thống (vốn không phân biệt được *thiếu tri thức* với *đồng nhất hóa*), thực nghiệm sử dụng **3 tập dữ liệu nhỏ có độ phân giải chẩn đoán cao**:

| Tập Dữ liệu | Quy mô Mẫu | Độ dài Sinh (max\_new\_tokens) | Số mẫu/Câu (K) | Mục tiêu Đo lường Chuyên biệt |
| :---- | :---- | :---- | :---- | :---- |
| **CoverageQA** | 105 câu | 64 | $K=16$ | Câu hỏi thực tế dưới xác định từ Wikidata (trung bình 28,7 đáp án đúng/câu) — tách bạch **độ phủ tri thức (Recall@16)** khỏi **ảo giác (Precision@16)**. |
| **PopQA-Stratified** | 200 câu (100 Head \+ 100 Tail) | 64 | $K=16$ | Lấy mẫu phân tầng top-20% (Head) và bottom-20% (Tail) theo độ phổ biến thực thể s\_pop — đo **tốc độ xói mòn tri thức đuôi dài (THR)** và **ảo giác hai chiều**. |
| **INFINITY-CHAT100** | 100 câu | 128 | $K=16$ | Tập chuẩn 100 truy vấn mở đa dạng từ WildChat — đo **sụp đổ thứ tự sinh (${H}_{order},{\rho }_{LTR}$)** và **đồng nhất hóa ý tưởng (Vendi Score)**. |

### **5.3. Hệ thống Thước đo Định lượng 3 Tầng**

#### **Tầng 1: Thước đo Tri thức Sự kiện & Độ chính xác**

**Multi-Answer Coverage (Recall@16 & Precision@16):** Trích xuất tập các thực thể duy nhất ${\hat{Y}}_{i}$ từ $K=16$ phản hồi cho câu hỏi $i$, đối chiếu chuẩn hóa với tập đáp án vàng ${Y}_{i}^{*}$ của CoverageQA:

$$Recall@16=\frac{1}{N}\sum\limits_{i=1}^{N}\frac{|{\hat{Y}}_{i}\cap {Y}_{i}^{*}|}{|{Y}_{i}^{*}|},Precision@16=\frac{1}{N}\sum\limits_{i=1}^{N}\frac{|{\hat{Y}}_{i}\cap {Y}_{i}^{*}|}{|{\hat{Y}}_{i}|}$$

*   
* **Tail-to-Head Retention Ratio ($THR$):** Đo mức suy giảm bất đối xứng trên thực thể hiếm so với thực thể phổ biến trên PopQA-Stratified.

#### **Tầng 2: Thước đo Đa dạng Phổ & Tách bạch Nội dung – Hình thức**

**Vendi Score bậc $q$ ($V{S}_{0.5},V{S}_{1},V{S}_{\infty }$):** Tính từ trị riêng chuẩn hóa \$\\lambda\_1, \\dots, \\lambda\_K\$ của ma trận tương đồng cosine $S/K$ trên không gian embedding BAAI/bge-m3:

$$V{S}_{q}=\exp \left({\frac{1}{1-q}\log \sum\limits_{m=1}^{K}{\lambda }_{m}^{q}}\right)$$

*   
  Trong đó bậc $q=0.5$ nhạy cảm cao với sự biến mất của các **ý tưởng/tri thức đuôi dài hiếm gặp**, còn $q\rightarrow \infty$ bắt hiện tượng **sao chép trùng lặp**.  
* **Đa dạng Từ vựng Độc lập Độ dài (MTLD):** Tách bạch đa dạng cấu trúc bề mặt khỏi đa dạng ngữ nghĩa sâu.

#### **Tầng 3: Thước đo Giải phẫu Quỹ đạo Nội tại dLLM ($t=1\rightarrow T$)**

* **Stepwise Mask Entropy (${H}_{t}$):** Entropy trung bình của phân phối logits trên tập vị trí còn bị đánh mặt nạ ${M}_{t}$ tại bước $t$:  
  \$\$H\_t \= \\frac{1}{\|\\mathcal{M}\_t\|} \\sum\_{j \\in \\mathcal{M}\_t} \\left( \-\\sum\_{v \\in \\mathcal{V}} P\_t(v \\mid x^{(t)}\_j) \\log\_2 P\_t(v \\mid x^{(t)}\_j) \\right)\$\$  
* **Early Boilerplate Lock-in Rate ($EBLR$):** Tỷ lệ token mở khóa trong pha đầu \$t \\in \[1, \\lfloor 0.25T \\rfloor\]\$ thuộc từ điển từ chức năng (stop-words, dấu câu) và cụm mở đầu khuôn mẫu.  
* **Unmasking Order Entropy (${H}_{order}$) & Left-to-Right Correlation (${\rho }_{LTR}$):** Tính trực tiếp từ ma trận bước mở khóa đầu tiên \$\\tau \\in \\{1,\\dots,T\\}^{K \\times L}\$.  
* **Self-Confirmation Confidence Drift ($\Delta {C}_{wrong}$):** Chênh lệch xác suất của thực thể sai giữa bước cuối $T$ và bước mở khóa ${t}_{unmask}$.

### **5.4. Ma trận Thực nghiệm và Kiểm định Thống kê**

| Mã Thí nghiệm | Mục tiêu Kiểm định | Mô hình & Cấu hình Chạy | Tập Dữ liệu | Chỉ số Đầu ra Chính |
| :---- | :---- | :---- | :---- | :---- |
| **Exp-1 (FM-1)** | Đo sụp đổ tri thức đa đáp án, xói mòn đuôi dài và khóa khung sớm hậu huấn luyện | Dream-7B (Base, Instruct) vs Qwen2.5-7B (Base, Instruct); LLaDA-8B vs Llama-3.1-8B (steps=64, $K=16$) | CoverageQA (105) PopQA (200) | Recall@16, Precision@16, THR, EBLR, đường cong ${H}_{t}$ |
| **Exp-2 (FM-2)** | Đo sụp đổ thứ tự giải mặt nạ phi tuyến và liên hệ với đồng nhất hóa câu hỏi mở | Dream-7B (Base, Instruct), LLaDA-8B (Base, Instruct), trích xuất ma trận ${\tau }_{k,j}$ ($K=16$) | INFINITY-CHAT100 (100) CoverageQA (105) | ${H}_{order}$, ${\rho }_{LTR}$, $V{S}_{0.5}$, $V{S}_{1}$, MTLD |
| **Exp-3 (FM-3)** | Đo hiệu ứng dây chuyền ảo giác hai chiều dưới áp lực nén bước khử nhiễu | Dream-7B-Instruct vs Qwen2.5-7B-Instruct; quét $T\in \{16,32,64,128\}$ ($K=8$) | PopQA (200) CoverageQA (105) | Hallucination Rate, Step-Compression Slope, $\Delta {C}_{wrong}$ |
| **Exp-4 (FM-4)** | Dựng đường biên Pareto của 4 chiến lược remasking tĩnh | Dream-7B & LLaDA-8B (Base & Instruct) qua origin, maskgit\_plus, topk\_margin, entropy | Tập gộp 150 câu (50 CovQA \+ 50 Tail \+ 50 Chat) | Đường biên Pareto: (Recall@16, $V{S}_{0.5}$) vs. (Precision@16, External PPL) |

**Tiêu chuẩn Kiểm định Thống kê:** Mọi chênh lệch giữa Base và Instruct hoặc giữa dLLM và AR đều được báo cáo kèm **khoảng tin cậy Bootstrap 95%** (1.000 lần lấy mẫu lại phân tầng) và kiểm định phi tham số ghép cặp (**Wilcoxon signed-rank test / permutation test**). Đối với đánh giá tính tương đương ngữ nghĩa trên câu hỏi mở (INFINITY-CHAT100), thực hiện kiểm tra chéo với người đánh giá trên tập mẫu phân tầng 150 phản hồi để báo cáo độ đồng thuận **Krippendorff's $\alpha$**, tuân thủ nghiêm ngặt quy định chống lỗi *M1 (LLM-only evaluation without validation)* của ARR.

## **6\. Hướng Mở rộng Can thiệp và Đóng góp Dự kiến cho Bài báo Hoàn chỉnh**

Để bài báo vượt qua ngưỡng Findings và cạnh tranh trực tiếp vị trí **ACL 2027 Main Conference**, kết quả chẩn đoán từ Exp-1 đến Exp-4 sẽ dẫn trực tiếp tới một đóng góp thuật toán gọn nhẹ, không cần huấn luyện lại (**Training-Free Inference-Time Mitigation**):

* **Đề xuất Kỹ thuật — *Phase-Aware Annealing-Margin Remasking (PA-AMR)*:** Tận dụng generation\_logits\_hook\_func trong kiến trúc Dream/LLaDA để tách quá trình khử nhiễu thành hai pha rõ rệt:  
  1. **Pha Khám phá Ý tưởng & Thực thể (\$t \\in \[1, \\lfloor 0.35T \\rfloor\]\$):** Áp dụng nhiệt độ phân phối cao hơn (${\tau }_{early}>1.0$) kết hợp nhiễu Gumbel có kiểm soát vào điểm số chọn vị trí unmasking và phạt nhẹ các token thuộc nhóm từ chức năng khuôn mẫu, phá vỡ sự khóa cứng cấu trúc sớm (khắc phục **FM-1** và **FM-2**).  
  2. **Pha Tinh chỉnh Cú pháp & Xác minh Sự kiện (\$t \\in (\\lfloor 0.35T \\rfloor, T\]\$):** Chuyển sang chiến lược chọn theo biên độ xác suất (topk\_margin) với nhiệt độ thấp (${\tau }_{late}\leq 0.3$), khóa chặt độ chính xác thực thể và ngăn chặn chuỗi ảo giác hai chiều (khắc phục **FM-3** và **FM-4**).  
* **Hệ thống Đối chứng (Baselines):** Đặt **PA-AMR** lên bàn cân Pareto trực tiếp với *Direct Sampling*, *Temperature Scaling*, *Verbalized Sampling (VS)* và 4 chiến lược remasking tĩnh mặc định.  
* **Tóm tắt 3 Đóng góp Cốt lõi của Bài báo:**  
  1. **Phát hiện Thực chứng đầu tiên về Knowledge Collapse trên dLLM:** Chứng minh có kiểm soát nhân quả qua cặp song sinh Dream-7B / Qwen2.5-7B và LLaDA-8B / Llama-3.1-8B cách thức và mức độ hậu huấn luyện làm xói mòn tri thức đa đáp án và thực thể đuôi dài trên mô hình khuếch tán.  
  2. **Giải phẫu Cơ chế Quỹ đạo Khử nhiễu:** Đưa ra hệ thước đo nội tại mới (${H}_{order},{\rho }_{LTR},EBLR,\Delta {C}_{wrong}$), vạch trần sự sụp đổ tính phi tuyến tính và cơ chế tự củng cố ảo giác hai chiều qua từng bước thời gian $t$.  
  3. **Giải pháp Điều tiết Động không cần Huấn luyện lại:** Đề xuất *PA-AMR* đẩy đường biên đánh đổi *Diversity (Recall@K / Vendi Score) vs. Factuality (Precision@K)* vượt trội các chiến lược tĩnh và bổ trợ hiệu quả cho *Verbalized Sampling*.

## **7\. Quản trị Rủi ro và Cổng Quyết định (Risk Management & Decision Gates)**

| Rủi ro Kỹ thuật / Học thuật | Dấu hiệu Nhận biết Sớm | Phương án Ứng phó & Chuyển hướng Tức thì |
| :---- | :---- | :---- |
| **Tín hiệu một FM yếu hoặc không có ý nghĩa thống kê** (ví dụ: ${\rho }_{LTR}$ không đổi giữa Base và Instruct) | Sau chạy Exp-1 & Exp-2 trên 100 mẫu đầu, $p>0.05$ hoặc khoảng tin cậy 95% chồng lấn hoàn toàn. | Không ép diễn giải; loại bỏ FM đó khỏi luận điểm chính và dồn toàn bộ dung lượng phân tích sâu vào 2 FM có tín hiệu rõ nhất (ví dụ tập trung vào **FM-1 \+ FM-4** hoặc **FM-3**). |
| **Nghẽn tốc độ sinh mẫu trên T4 16 GB** | Thời gian sinh 16 mẫu/câu vượt quá 90 giây khiến tổng thời gian vượt 15 GPU-giờ. | Giảm số mẫu lấy mẫu xuống $K=8$ cho tập lớn (PopQA) và giữ $K=16$ cho tập đa đáp án (CoverageQA), đồng thời bật cơ chế tái sử dụng KV-cache khối (*Fast-dLLM / dLLM-Cache*). |
| **Sai lệch do lượng tử hóa 4-bit NF4** | FVU hoặc Perplexity của bản 4-bit lệch nhiều so với bản fp16. | Chạy kiểm tra đối chiếu độ nhạy trên tập con 50 câu với bản fp16 (batch\_size=1, max\_new\_tokens=64) để chứng minh xu hướng sụp đổ giữa Base và Instruct bất biến theo độ chính xác số học. |
| **Yêu cầu Liêm chính & Hành chính ARR 01/2027** | Nguy cơ desk reject do thiếu thủ tục OpenReview hoặc lỗi trích dẫn. | Hoàn tất xác nhận *Designated Service Contributor* và ORCID trước hạn; kiểm toán 100% file .bib qua công cụ đối chiếu DBLP/ACL Anthology; viết mục *Limitations* phân tích thẳng thắn phạm vi quy mô 7B–8B và ảnh hưởng của lượng tử hóa 4-bit. |

