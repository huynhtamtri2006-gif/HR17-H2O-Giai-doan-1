# HR17 — Báo cáo giai đoạn 1

Ánh xạ chuyển động người sang robot: kiểm chứng thu gọn H2O trên Unitree H1.
Ngày chốt kết quả: 24/11/2026. Bộ này chuẩn bị cho repository PRIVATE để nộp học phần.

## Mở nhanh báo cáo và slide

Hai file có thể chỉnh sửa nằm trong thư mục [BAO_CAO_VA_THUYET_TRINH](BAO_CAO_VA_THUYET_TRINH/):

- [Báo cáo Word — cập nhật 24/11/2026](BAO_CAO_VA_THUYET_TRINH/Bao_cao_Giai_doan_1_HR17_Hoan_thien_FINAL_Cap_nhat_24_11_2026.docx)
- [PowerPoint — cập nhật 24/11/2026](BAO_CAO_VA_THUYET_TRINH/HR17_Thuyet_trinh_Giai_doan_1_FINAL_Cap_nhat_24_11_2026.pptx)

Đây là hai bản sao trùng nội dung với file tương ứng trong docs/; các bản PDF vẫn ở docs/.

## Cấu trúc

- BAO_CAO_VA_THUYET_TRINH/: hai file Word và PowerPoint chính để mở nhanh.
- docs/: báo cáo Word/PDF và PowerPoint/PDF.
- demo/: video id01 và khung hình từ dữ liệu mô phỏng.
- notebooks/: notebook lịch sử; bỏ output và metadata ở bản sao, giữ nguyên code.
- src/: script thử nghiệm, đánh giá, khôi phục.
- results/: CSV, biểu đồ và bản kiểm chứng.
- third_party/: README và giấy phép mã kế thừa.

## Kết quả và giới hạn

Checkpoint được chọn sau tinh chỉnh đạt 2,26 s trên id01 so với đối chứng 1,66 s;
cả hai dùng Kd cổ chân = 1. id02: 1,46 → 2,14 s; ood03 giảm chất lượng.
Tất cả các lượt trong phép đánh giá mới vẫn reset sớm do gravity_x.
Năm lượt đánh giá cố định đạt 2,26 s không phải năm seed huấn luyện độc lập.
Hai lần train lại cho 2,26 s và 1,40 s; chưa chứng minh train tái lập ổn định.
Chưa bám trọn clip, chưa có camera/VR trực tiếp hoặc đo độ trễ đầu–cuối.
Video phát đúng thời gian mô phỏng, không nối episode.

## Chạy lại và tài nguyên để ngoài repo

Đây là bộ tài liệu và mã nộp bài, CHƯA phải môi trường chạy được ngay khi clone.
Không Run all notebook chứa ô lịch sử. Đọc hướng dẫn trước khi chọn ô.
Notebook đánh giá hiện có tên HR17_Danh_gia_6_Motion_ID_OOD_FINAL.ipynb;
hướng dẫn khôi phục cũ dùng tên chưa có hậu tố FINAL.

Mã nền: https://github.com/LeCAR-Lab/human2humanoid
Commit: 750f1fa052641f0fde43669d50cb4e407dabe6c8.
Người chạy cần tải mã nền; src/ không thay thế toàn bộ mã nền.
Đường dẫn /content trong script là đường dẫn Google Colab.

Không kèm Isaac Gym, AMASS, motion pickle, tập trích sáu clip, SMPL/SMPL-X,
checkpoint .pt hoặc ZIP thô. Người chạy phải có đúng nguồn và quyền sử dụng.
Isaac Gym: https://developer.nvidia.com/isaac-gym/download
AMASS: https://amass.is.tue.mpg.de/license.html
Tên gói và mã kiểm tra trong results/ dùng truy vết bản gốc giữ trên máy,
không có nghĩa các gói đó được phân phối qua repo này.

requirements_upstream.txt là yêu cầu mã nền, không phải lockfile đầy đủ Colab.
Phiên đã dùng: Python 3.8.20, PyTorch 2.4.1+cu121, NumPy 1.23.0,
Isaac Gym Preview 4 và Tesla T4.

## Nguồn và đóng góp

H2O/OmniH2O: LeCAR-Lab/human2humanoid; kế thừa Legged Gym, RSL RL, PHC
và tác giả được ghi trong README/giấy phép gốc. Script có thể chứa mã kế thừa.
Đóng góp nhóm: khôi phục môi trường, tổ chức thử nghiệm thu gọn, huấn luyện,
đối chứng, kiểm chứng số liệu, demo và báo cáo; không nhận toàn bộ thuật toán nền là tự viết.

Giữ giấy phép trong third_party/. Không áp giấy phép mới cho toàn bộ tài nguyên.
Mã nền ghi CC BY-NC 4.0 cùng giấy phép kế thừa. Private không thay thế quyền phân phối.
Báo cáo có họ tên/MSSV: rà soát trước khi chuyển Public.
