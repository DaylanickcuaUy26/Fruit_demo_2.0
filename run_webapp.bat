@echo off
echo ========================================
echo    HE THONG NHAN DIEN QUA AI
echo ========================================
echo.
echo Dang kiem tra mo hinh...
if not exist "fruit_classifier_mobilenetv2.h5" (
    echo.
    echo ❌ Khong tim thay mo hinh da huan luyen!
    echo Vui long chay "python train_model.py" truoc.
    echo.
    pause
    exit /b 1
)

echo ✅ Mo hinh da san sang!
echo.
echo Dang kiem tra thu vien...
python -c "import flask, tensorflow, cv2" 2>nul
if errorlevel 1 (
    echo.
    echo ❌ Thieu thu vien! Dang cai dat...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ❌ Loi cai dat thu vien!
        pause
        exit /b 1
    )
)

echo ✅ Thu vien da san sang!
echo.
echo 🚀 Khoi dong web application...
echo 📱 Trinh duyet se tu dong mo trong vai giay...
echo.
echo Nhan Ctrl+C de dung chuong trinh
echo.

python app.py

pause 