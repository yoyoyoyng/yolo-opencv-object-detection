# Labels

이 폴더에는 사용자가 Ubuntu에서 직접 생성한 YOLO 라벨 파일을 넣습니다.

기존 작업 폴더가 다음 위치라면:

```bash
~/Downloads/H13_smartcar_dataset_labeling_ready_v2/labels
```

저장소 루트에서 아래처럼 복사합니다.

```bash
cp -r ~/Downloads/H13_smartcar_dataset_labeling_ready_v2/labels/train/* labels/train/
cp -r ~/Downloads/H13_smartcar_dataset_labeling_ready_v2/labels/val/* labels/val/
cp -r ~/Downloads/H13_smartcar_dataset_labeling_ready_v2/labels/test/* labels/test/
```
