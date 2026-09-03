import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO

if __name__ == "__main__":
    # 论文模型配置：DySnakeConv + EUCB-SC + CoordAtt
    model = YOLO("./ultralytics/cfg/models/11puls/yolo11-dysnake-EUCB-SC-CoordAtt.yaml")

    model.train(
        data="./datasets/data.yaml",  # 替换为你的数据集 data.yaml 路径
        cache=False,
        imgsz=640,
        epochs=200,
        batch=16,
        close_mosaic=0,  # 最后多少个 epoch 关闭 mosaic 数据增强，0 表示全程开启
        workers=4,  # Windows 下卡住可尝试设为 0
        optimizer="SGD",
        # device="0",  # 指定显卡
        # resume=True,  # 断点续训（YOLO 初始化时选 last.pt）
        # amp=False,  # loss 出现 nan 可关闭 AMP
        project="runs/train",
        name="exp",
        seed=123,
    )
