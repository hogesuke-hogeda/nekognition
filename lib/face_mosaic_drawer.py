from abc import ABC, abstractmethod
from mypy_boto3_rekognition.type_defs import FaceDetailTypeDef
from PIL import Image, ImageDraw


class IFaceMosaicDrawer(ABC):
    @abstractmethod
    def apply_mosaic(self, image: Image.Image, face_details: list[FaceDetailTypeDef], mosaic_size: int) -> Image.Image:
        pass


class EllipseFaceMosaicDrawer(IFaceMosaicDrawer):
    def apply_mosaic(self, image: Image.Image, face_details: list[FaceDetailTypeDef], mosaic_size: int) -> Image.Image:
        """顔が検出された位置に楕円形のモザイクを描画する"""

        # 顔が検出されなかった場合は元の画像を返す
        if len(face_details) == 0:
            return image

        mosaic_layer = Image.new("RGB", image.size)  # モザイク処理を適用した領域のレイヤー
        mask = Image.new("L", image.size, 0)  # モザイクレイヤーと元画像を合成する用のマスク
        draw = ImageDraw.Draw(mask)

        for face_detail in face_details:
            if "BoundingBox" in face_detail:
                box = face_detail["BoundingBox"]
                # 対象領域を計算
                image_width, image_height = image.size
                left = int(box["Left"] * image_width)
                top = int(box["Top"] * image_height)
                right = int(left + box["Width"] * image_width)
                bottom = int(top + box["Height"] * image_height)

                # モザイク処理を適用する領域を切り抜き
                region = image.crop((left, top, right, bottom))
                region = region.resize(
                    (max(1, (right - left) // mosaic_size),
                        max(1, (bottom - top) // mosaic_size)),
                )
                region = region.resize((right - left, bottom - top))

                # モザイク処理した領域をモザイクレイヤーに貼り付け
                mosaic_layer.paste(region, (left, top, right, bottom))

                # 楕円形のマスクを作成
                draw.ellipse(
                    [(left, top), (right, bottom)],
                    fill=255
                )

        # マスクを使用してモザイクレイヤーを適用
        result_image = Image.composite(mosaic_layer, image, mask)
        return result_image
