import numpy as np
from PIL import Image
import struct
import os
import zlib

class BMPCompressor:
    def __init__(self, quantization_level=32, block_size=8, quality=50):
        self.quantization_level = quantization_level
        self.block_size = block_size
        self.quality = quality
        # JPEG 標準量化矩陣（擴展版）
        self.quant_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ]) * (quality / 50.0) / 8.0  # 根據品質參數調整量化強度
    
    def read_bmp(self, filename):
        """讀取 BMP 檔案"""
        with Image.open(filename) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)

    def compress(self, input_file, output_file):
        """壓縮 BMP 檔案"""
        # 讀取原始圖片
        original_img = self.read_bmp(input_file)
        
        # 分塊處理
        height, width = original_img.shape[:2]
        processed_img = np.zeros_like(original_img, dtype=np.float32)
        
        # 對每個顏色通道分別處理
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                for c in range(3):  # RGB channels
                    block = original_img[i:i+self.block_size, j:j+self.block_size, c].astype(np.float32)
                    # DCT 轉換和量化
                    processed_block = self._process_block(block)
                    processed_img[i:i+self.block_size, j:j+self.block_size, c] = processed_block
        
        # 將處理後的圖片轉回 uint8
        processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
        
        # 使用 zlib 進行額外的無損壓縮
        compressed_data = zlib.compress(processed_img.tobytes(), level=9)
        
        # 儲存壓縮資料
        self._save_compressed(compressed_data, original_img.shape, output_file)
        
        # 計算壓縮率
        original_size = os.path.getsize(input_file)
        compressed_size = os.path.getsize(output_file)
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        # 計算 MSE
        decompressed = self.decompress(output_file)
        mse = self._calculate_mse(original_img, decompressed)
        
        return compression_ratio, mse

    def _process_block(self, block):
        """處理單個區塊：DCT 轉換和量化"""
        # 對區塊進行簡單的 DCT 類似的變換
        mean_val = np.mean(block)
        centered = block - mean_val
        
        # 使用量化矩陣進行量化
        quantized = np.round(centered / (self.quant_matrix[:self.block_size, :self.block_size] + 1))
        
        # 反量化和重建
        reconstructed = quantized * (self.quant_matrix[:self.block_size, :self.block_size] + 1)
        reconstructed += mean_val
        
        return reconstructed

    def _save_compressed(self, compressed_data, shape, output_file):
        """儲存壓縮後的資料"""
        with open(output_file, 'wb') as f:
            # 儲存圖片尺寸和參數
            header = struct.pack('IIII', shape[0], shape[1], shape[2], 
                               self.quantization_level)
            f.write(header)
            # 儲存壓縮後的資料
            f.write(compressed_data)

    def decompress(self, compressed_file):
        """解壓縮檔案"""
        with open(compressed_file, 'rb') as f:
            # 讀取 header
            height, width, channels, quant_level = struct.unpack('IIII', f.read(16))
            
            # 讀取壓縮資料並解壓縮
            compressed_data = f.read()
            decompressed_data = zlib.decompress(compressed_data)
            
            # 重建圖片陣列
            img_array = np.frombuffer(decompressed_data, dtype=np.uint8)
            img_array = img_array.reshape((height, width, channels))
            
            return img_array

    def _calculate_mse(self, original, decompressed):
        """計算均方誤差 (MSE)"""
        return np.mean((original.astype(np.float32) - decompressed.astype(np.float32)) ** 2)

def main():
    # 使用範例
    compressor = BMPCompressor(quantization_level=32, block_size=8, quality=50)
    
    input_file = "./img/BaboonRGB.bmp"
    compressed_file = "compressed.bin"
    
    # 壓縮並取得統計資料
    compression_ratio, mse = compressor.compress(input_file, compressed_file)
    
    print(f"壓縮率: {compression_ratio:.2f}%")
    print(f"均方誤差 (MSE): {mse:.2f}")
    
    # 解壓縮
    decompressed = compressor.decompress(compressed_file)
    
    # 儲存解壓縮後的圖片
    Image.fromarray(decompressed).save("decompressed.bmp")

if __name__ == "__main__":
    main()