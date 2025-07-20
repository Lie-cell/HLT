import jiwer

def calculate_cer_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()] 
        
        # 跳过前三行
        if len(lines) > 3:
            lines = lines[3:]
        
        cer_values = []
        current_unit = []  # 临时存储当前单元的行
        
        for line in lines:
            if not line:  # 遇到空行，处理当前单元
                if len(current_unit) == 3: 
                    ref_line = current_unit[0]
                    origin_line = current_unit[2]  # 第三行是Origin_predicted
                    
                    # 提取文本内容（冒号后的部分）
                    ref_text = ref_line.split(':', 1)[1].strip() if ':' in ref_line else ref_line
                    origin_text = origin_line.split(':', 1)[1].strip() if ':' in origin_line else origin_line
                    
                    if ref_text:  # 避免空文本
                        cer = jiwer.cer(ref_text, origin_text)
                        cer_values.append(cer)
                        unit_num = len(cer_values)
                        print(f"单元 {unit_num}: CER = {cer:.4f}")
                        print(f"  参考: '{ref_text}'")
                        print(f"  预测: '{origin_text}'")
                current_unit = []  # 重置当前单元
            elif line:  # 非空行添加到当前单元
                current_unit.append(line)
        
        if current_unit and len(current_unit) == 3:
            ref_line = current_unit[0]
            origin_line = current_unit[2]
            ref_text = ref_line.split(':', 1)[1].strip() if ':' in ref_line else ref_line
            origin_text = origin_line.split(':', 1)[1].strip() if ':' in origin_line else origin_line
            if ref_text:
                cer = jiwer.cer(ref_text, origin_text)
                cer_values.append(cer)
                unit_num = len(cer_values)
                print(f"单元 {unit_num}: CER = {cer:.4f}")
                print(f"  参考: '{ref_text}'")
                print(f"  预测: '{origin_text}'")
        
        # 计算平均CER
        if cer_values:
            avg_cer = sum(cer_values) / len(cer_values)
            print(f"\n计算完成！共处理 {len(cer_values)} 个单元")
            print(f"最终平均 CER: {avg_cer:.4f}")
            return avg_cer
        else:
            print("未找到有效单元数据！")
            return None
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    file_path = "vqvae_test_results_optimize.txt"  
    calculate_cer_from_file(file_path)