# scripts/05_promote_model.py  (ไฟล์ใหม่ที่ CD เรียกใช้)
import sys
 
from mlflow import MlflowClient
 
MODEL_NAME = "wine-classifier-prod"
 
 
def promote(src="staging", dst="champion"):
    client = MlflowClient()
    # ถ้าไม่มี @staging (โมเดลไม่ผ่านเกณฑ์ 0.95 เลยไม่ถูก register) จะ error -> job แดง -> ไม่ deploy
    mv = client.get_model_version_by_alias(MODEL_NAME, src)
    client.set_registered_model_alias(MODEL_NAME, dst, mv.version)
    print(f"@{dst} -> {MODEL_NAME} version {mv.version}")
 
 
if __name__ == "__main__":
    # rollback: python scripts/05_promote_model.py <version>  (ชี้ champion กลับเวอร์ชันเก่า)
    if len(sys.argv) > 1:
        MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", sys.argv[1])
        print(f"Rolled back @champion -> version {sys.argv[1]}")
    else:
        promote()