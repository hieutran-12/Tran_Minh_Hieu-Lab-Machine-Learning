# display.py — hàm hiển thị dùng chung cho toàn giáo trình
import cv2

windown_name2 = 'Supervision'
Display_Width = 1280

def show_frame1(frame,windows_name:str = windown_name2,wait : int = 1 ):
    """Hiện frame lên cửa sổ. Trả về False nếu người dùng bấm Q/ESC (muốn thoát).

    wait=1  -> dùng cho video (hiện liên tục, không chặn)
    wait=0  -> dùng cho ảnh tĩnh (dừng lại chờ bấm phím bất kỳ)
    """
    h,w = frame.shape[:2]
    if w > Display_Width:
        scale = Display_Width / w
        frame = cv2.resize(frame,(int(w * scale), int(h * scale)))
    
    cv2.imshow(windows_name,frame)
    key = cv2.waitKey(wait) & 0xFF
    
    if key in (ord('q'),ord('Q'),27):
        return False
    return True
def close_windows():
    cv2.destroyAllWindows()