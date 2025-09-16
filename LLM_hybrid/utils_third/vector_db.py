# utils/vectordb.py
from pathlib import Path
import os, shutil, tempfile
import gdown


# 구글 드라이브 링크 들어가서 그 안에 있는거를 통째로 가져와서, 그거를 utils/chroma_db 폴더에 넣기
def download_drive_folder_to_chroma_db(folder_url: str, target_dir: Path):
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        # gdown.download_folder 대신 다른 방법 시도
        success = False

        # 방법 1: 공개 링크로 폴더 다운로드 시도
        try:
            # Django 환경에서 gdown 설정
            import os

            os.environ["GDOWN_USE_COOKIES"] = "False"
            gdown.download_folder(
                url=folder_url, output=td, quiet=True, use_cookies=False
            )
            success = True
        except Exception:
            pass

        # 방법 2: 폴더 ID를 직접 사용하여 시도 (공개 링크)
        if not success:
            try:
                folder_id = folder_url.split("/")[-1]
                public_folder_url = (
                    f"https://drive.google.com/drive/folders/{folder_id}"
                )
                gdown.download_folder(
                    url=public_folder_url, output=td, quiet=True, use_cookies=False
                )
                success = True
            except Exception:
                pass

        # 방법 3: ZIP 파일로 다운로드 시도 (공개 링크)
        if not success:
            try:
                folder_id = folder_url.split("/")[-1]
                zip_url = f"https://drive.google.com/drive/folders/{folder_id}?export=download"
                zip_path = Path(td) / "folder.zip"
                gdown.download(zip_url, str(zip_path), quiet=True)

                import zipfile

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(td)
                success = True
            except Exception:
                pass

        # 방법 4: Django 환경에서 안전한 다운로드
        if not success:
            try:
                import requests

                folder_id = folder_url.split("/")[-1]
                download_url = (
                    f"https://drive.google.com/uc?export=download&id={folder_id}"
                )

                session = requests.Session()
                response = session.get(download_url, stream=True)

                if response.status_code == 200:
                    zip_path = Path(td) / "folder.zip"
                    with open(zip_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    import zipfile

                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(td)
                    success = True
            except Exception:
                pass

        if not success:
            raise RuntimeError("모든 다운로드 방법이 실패했습니다.")

        # 다운로드된 내용 확인
        downloaded_items = os.listdir(td)
        entries = [Path(td) / name for name in downloaded_items]
        src_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(td)

        # 재귀적으로 모든 파일과 폴더 복사
        def copy_recursive(src, dst):
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        for p in src_root.iterdir():
            dst = target_dir / p.name
            copy_recursive(p, dst)

    if not (target_dir / "chroma.sqlite3").exists():
        raise RuntimeError(f"'chroma.sqlite3'가 없습니다: {target_dir}")


def create_chroma_db():
    HERE = Path(__file__).resolve().parent
    FOLDER_URL = "https://drive.google.com/drive/folders/1wp-Cy05VKJi1kYAku8AkJylO4i0hqdzN?usp=drive_link"
    target_path = HERE / "chroma_db"

    download_drive_folder_to_chroma_db(FOLDER_URL, target_path)
