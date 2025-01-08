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
        self.quant_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ]) * (quality / 50.0) / 8.0 
    
    def read_bmp(self, filename):
        with Image.open(filename) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)

    def compress(self, input_file, output_file):
        original_img = self.read_bmp(input_file)
        
        height, width = original_img.shape[:2]
        processed_img = np.zeros_like(original_img, dtype=np.float32)
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                for c in range(3):  # RGB channels
                    block = original_img[i:i+self.block_size, j:j+self.block_size, c].astype(np.float32)
                    processed_block = self._process_block(block)
                    processed_img[i:i+self.block_size, j:j+self.block_size, c] = processed_block
        
        processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)
        
        compressed_data = zlib.compress(processed_img.tobytes(), level=9)
        
        self._save_compressed(compressed_data, original_img.shape, output_file)
        
        original_size = os.path.getsize(input_file)
        compressed_size = os.path.getsize(output_file)
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        decompressed = self.decompress(output_file)
        mse = self._calculate_mse(original_img, decompressed)
        
        return compression_ratio, mse

    def _process_block(self, block):
        mean_val = np.mean(block)
        centered = block - mean_val
        
        quantized = np.round(centered / (self.quant_matrix[:self.block_size, :self.block_size] + 1))
        
        reconstructed = quantized * (self.quant_matrix[:self.block_size, :self.block_size] + 1)
        reconstructed += mean_val
        
        return reconstructed

    def _save_compressed(self, compressed_data, shape, output_file):
        with open(output_file, 'wb') as f:
            header = struct.pack('IIII', shape[0], shape[1], shape[2], 
                               self.quantization_level)
            f.write(header)
            f.write(compressed_data)

    def decompress(self, compressed_file):
        with open(compressed_file, 'rb') as f:
            height, width, channels, quant_level = struct.unpack('IIII', f.read(16))
            
            compressed_data = f.read()
            decompressed_data = zlib.decompress(compressed_data)
            
            img_array = np.frombuffer(decompressed_data, dtype=np.uint8)
            img_array = img_array.reshape((height, width, channels))
            
            return img_array

    def _calculate_mse(self, original, decompressed):
        """計算均方誤差 (MSE)"""
        return np.mean((original.astype(np.float32) - decompressed.astype(np.float32)) ** 2)

def main():
    compressor = BMPCompressor(quantization_level=32, block_size=8, quality=50)
    
    input_file = "./img/BaboonRGB.bmp"
    compressed_file = "compressed.bin"
    
    compression_ratio, mse = compressor.compress(input_file, compressed_file)
    
    print(f"壓縮率: {compression_ratio:.2f}%")
    print(f"均方誤差 (MSE): {mse:.2f}")
    
    decompressed = compressor.decompress(compressed_file)
    
    Image.fromarray(decompressed).save("decompressed.bmp")

if __name__ == "__main__":
    main()