"""Đóng ZIP các cặp đầy đủ, kiểm tra ZIP rồi xóa ảnh lẻ và cache tải xuống."""
import json
from pathlib import Path
import shutil
import zipfile

from tqdm import tqdm


def image_index(folder):
    images = {}
    for path in folder.rglob('*'):
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
            key = path.relative_to(folder).with_suffix('').as_posix()
            if key in images:
                raise ValueError(f'Trùng tên ảnh: {key}')
            images[key] = path
    return images


def package_and_clean(root):
    root = Path(root).resolve(strict=True)
    archive = root/'AnimeFaceColorization_paired.zip'
    sketches, colors = image_index(root/'sketch'), image_index(root/'reference')
    paired = sorted(sketches.keys() & colors.keys())
    if not paired:
        raise ValueError('Không có cặp ảnh đầy đủ; dừng, không xóa dữ liệu.')
    keep = [index[key] for key in paired for index in [sketches, colors]]
    if (root/'README.md').is_file():
        keep.append(root/'README.md')
    remove = [sketches[key] for key in sketches.keys()-colors.keys()]
    remove += [colors[key] for key in colors.keys()-sketches.keys()]
    # Kiểm tra toàn bộ đường dẫn trước khi có thao tác xóa.
    for path in keep + remove + [archive, root/'.cache']:
        path.resolve().relative_to(root)
    snapshot = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in [*sketches.values(), *colors.values()]}
    summary = dict(pairs=len(paired), unpaired_removed=len(remove), source=str(root))
    metadata = {'dataset_summary.json', 'excluded_files.txt'}

    if not archive.exists():
        partial = archive.with_suffix('.zip.partial')
        partial.resolve().relative_to(root)
        # JPEG/PNG đã nén sẵn: ZIP_STORED đóng gói nhanh, không làm giảm chất lượng.
        with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zip_file:
            for path in tqdm(keep, desc='Đóng gói ZIP', unit='file'):
                zip_file.write(path, path.relative_to(root).as_posix())
            zip_file.writestr('dataset_summary.json', json.dumps(summary, indent=2))
            zip_file.writestr('excluded_files.txt', '\n'.join(sorted(path.relative_to(root).as_posix() for path in remove)))
        partial.replace(archive)

    print('Kiểm tra danh sách, kích thước và CRC của toàn bộ ZIP...', flush=True)
    with zipfile.ZipFile(archive) as zip_file:
        expected = {path.relative_to(root).as_posix(): path.stat().st_size for path in keep}
        entries = zip_file.infolist()
        if len(entries) != len(expected)+len(metadata) or set(zip_file.namelist()) != set(expected)|metadata:
            raise ValueError('Nội dung ZIP không khớp dữ liệu hiện tại; chưa xóa file nào.')
        if any(zip_file.getinfo(name).file_size != size for name,size in expected.items()):
            raise ValueError('Kích thước file trong ZIP không khớp; chưa xóa file nào.')
        bad = zip_file.testzip()
        if bad:
            raise ValueError(f'ZIP lỗi CRC tại {bad}; chưa xóa file nào.')

    current = [*image_index(root/'sketch').values(), *image_index(root/'reference').values()]
    if {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in current} != snapshot:
        raise RuntimeError('Dữ liệu thay đổi khi đóng ZIP. Dừng tải dataset rồi chạy lại; chưa xóa file nào.')
    for path in tqdm(remove, desc='Xóa ảnh thiếu cặp', unit='file'):
        path.unlink()
    cache = root/'.cache'
    if cache.exists():
        cache.resolve().relative_to(root)
        shutil.rmtree(cache)
    for folder in [root/'sketch', root/'reference']:
        for directory in sorted((p for p in folder.rglob('*') if p.is_dir()), key=lambda p:len(p.parts), reverse=True):
            if not any(directory.iterdir()):
                directory.rmdir()
    print(f'Hoàn tất: {archive}\nGiữ {len(paired):,} cặp; xóa {len(remove):,} ảnh lẻ và cache.', flush=True)
    return archive


if __name__ == '__main__':
    destination = Path('D:/AnimeFaceColorization')
    # Kiểm tra quyền trước khi quét/đóng gói dữ liệu lớn.
    try:
        probe = destination/'.package_write_check'
        with probe.open('w'):
            pass
        probe.unlink()
    except PermissionError:
        raise SystemExit('Windows chặn ghi ổ D. Chạy script bằng PowerShell Administrator.')
    package_and_clean(destination)
