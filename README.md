# Diffusion Collapse

Repo khảo sát hiện tượng đồng nhất hóa đầu ra và suy giảm tri thức sau instruction tuning. Pipeline so sánh Dream-v0-7B (discrete diffusion) với Qwen2.5-7B (autoregressive) trên CoverageQA, PopQA và INFINITY-CHAT, với cả checkpoint Base lẫn Instruct.

## Cấu trúc và pipeline

- [Đề xuất nghiên cứu](research_propose.md), [thẻ thí nghiệm FM-1 đến FM-4](docs/r0_experiment_cards.md) và [cấu hình R0](configs/r0_protocol.yaml) mô tả giả thuyết, bộ dữ liệu, tham số và các cổng kiểm chứng R0–R7.
- [Chuẩn bị dữ liệu](scripts/prepare_datasets.py) tạo các split cố định trong `data/frozen/`. Hướng dẫn lấy CoverageQA nằm trong [data/raw](data/raw/CoverageQA_README.md).
- [Benchmark](scripts/run_benchmark.py) sinh mẫu và ghi quỹ đạo giải mã. [Kịch bản R5](scripts/run_r5_pilot.sh) chạy đối chứng Base/Instruct, khảo sát số bước khử nhiễu và so sánh các chiến lược remasking.
- [Đánh giá](scripts/evaluate_run.py) dùng [các metric](src/metrics/) để đo độ bao phủ kiến thức, đa dạng đầu ra, độ trôi chảy và thứ tự mở khóa token. [Phân tích R5](scripts/analyze_r5_pilot.py) tổng hợp kết quả theo từng failure mode.

## Kết quả pilot R5

Pilot so sánh Base/Instruct trên 100 câu hỏi với 8 mẫu mỗi câu. Khi giảm số bước khử nhiễu từ 64 xuống 16 trên PopQA, tỷ lệ ảo giác quan sát được tăng từ 0,669 lên 0,744. Thử nghiệm remasking cũng cho thấy sự đánh đổi giữa độ bao phủ và độ trôi chảy. Xem [bản tổng kết R5](results/r5_pilot/R5_DECISION_MEMO.md) và [kết quả đánh giá tổng hợp](results/r5_pilot/r5_eval_report.json).

Log, dự đoán từng mẫu, quỹ đạo token và bản báo cáo đầy đủ được lưu local. Sau khi có dữ liệu đóng băng và đầu ra các lượt chạy R5, chạy `python3 scripts/generate_r5_report_artifacts.py` để tạo hình và thống kê chi tiết. Nếu đã có `docs/R5_PILOT_FULL_REPORT.md` ở máy local, dùng `bash scripts/build_report_pdf.sh` với Pandoc và `uvx` để xuất HTML và PDF.
