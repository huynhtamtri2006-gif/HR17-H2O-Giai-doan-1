# HR17 — Báo cáo giai đoạn 1 — Nhóm 17

Đề tài: Ánh xạ chuyển động người sang robot — kiểm chứng thu gọn cấu hình H2O trên robot Unitree H1.

- Lớp: 15DHTH07.
- Giảng viên hướng dẫn: Nguyễn Tuấn Anh.
- Thành viên: Cao Minh Luân và Huỳnh Tâm Trí.
- Dữ liệu thực nghiệm được tổng hợp đến ngày 24/11/2026.
- Báo cáo và slide được biên tập, đồng bộ ngày 25/11/2026.

Repository công khai phục vụ lưu trữ, xem và tải tài liệu báo cáo giai đoạn 1.

## Báo cáo và slide mới nhất

Các tài liệu được lưu trong thư mục [docs](docs/).

- [Báo cáo Word](<docs/BÁO CÁO GIAI ĐOẠN 1 - NHÓM 17.docx>)
- [Slide PowerPoint](<docs/SLIDE THUYẾT TRÌNH GIAI ĐOẠN 1 - NHÓM 17.pptx>)

Các bản PDF, kịch bản thuyết trình và bộ câu hỏi phản biện được đặt trong thư mục docs nếu đã tải lên.

## Tải toàn bộ bộ tài liệu

Mở [Releases](https://github.com/huynhtamtri2006-gif/HR17-H2O-Giai-doan-1/releases), chọn bản phát hành “Bộ tài liệu nộp giai đoạn 1 - Nhóm 17” và tải file HocMay_GD1_Nhóm17_15DHTH07.zip trong mục Assets.

Giải nén ZIP để xem toàn bộ báo cáo, slide, demo, notebook, mã nguồn, cấu hình, checkpoint và bằng chứng thực nghiệm.

Hai mục “Source code (zip)” và “Source code (tar.gz)” do GitHub tự tạo không phải ZIP tổng hợp mà nhóm đính kèm.

## Phạm vi và kết quả chính

- Thực hiện trong mô phỏng Isaac Gym, dùng chuyển động tham chiếu lưu sẵn.
- Huấn luyện PPO thu gọn với ba seed.
- Khảo sát khối lượng, ma sát và độ trễ khi giữ policy cố định.
- Đánh giá sáu chuyển động thuộc và ngoài bộ huấn luyện.
- Checkpoint tinh chỉnh đạt thời gian trước reset 2,26 giây trên id01, so với đối chứng 1,66 giây trong cùng điều kiện so sánh.
- Bản mới cải thiện trên một số clip nhưng vẫn reset sớm; chưa bám ổn định toàn chuyển động.

**Kết luận:** Chưa tái lập được tuyên bố định lượng theo cùng tập dữ liệu và giao thức của bài báo. Chưa triển khai điều khiển người–robot trực tiếp hoặc đo độ trễ đầu–cuối.

## Cấu trúc repository

- docs/: báo cáo, slide và tài liệu hỗ trợ thuyết trình.
- demo/: video và hình minh họa từ dữ liệu mô phỏng.
- notebooks/: notebook thực hiện và khôi phục môi trường.
- src/: mã nguồn hỗ trợ thử nghiệm, đánh giá và khôi phục.
- results/: bảng số liệu, biểu đồ và kết quả kiểm chứng.
- third_party/: thông tin nguồn và giấy phép mã kế thừa.

Bộ tài liệu tổng hợp dung lượng lớn được đính kèm trong Releases.
