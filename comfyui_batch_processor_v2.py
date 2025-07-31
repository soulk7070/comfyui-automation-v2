import json
import requests
import websocket
import uuid
import time
import os
import re
import random
from datetime import datetime
import logging
from pathlib import Path

class ComfyUIBatchProcessorV2:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'batch_process_v2.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # --- PERUBAHAN UTAMA DIMULAI DI SINI ---
        self.workflows = self._discover_workflows()

    def _discover_workflows(self):
        """Secara otomatis memindai folder 'workflows' dan mendaftarkan semua file .json."""
        workflows_dir = Path('workflows')
        if not workflows_dir.is_dir():
            self.logger.error(f"Folder workflows tidak ditemukan di '{workflows_dir.resolve()}'")
            return {}

        discovered = {}
        self.logger.info("Mendeteksi workflow yang tersedia...")
        for workflow_file in workflows_dir.glob('*.json'):
            # Ambil nama file tanpa ekstensi .json sebagai nama kunci
            workflow_name = workflow_file.stem
            discovered[workflow_name] = str(workflow_file)
            self.logger.info(f" -> Ditemukan: '{workflow_name}' (dari file: {workflow_file.name})")
        
        if not discovered:
            self.logger.warning("PERINGATAN: Tidak ada workflow .json yang ditemukan di dalam folder 'workflows'.")
        
        return discovered
    # --- PERUBAHAN UTAMA SELESAI ---

    def parse_prompt_file(self, filepath):
        """Parser untuk format [∆{workflow}•jumlah∆ ¥prompt¥]"""
        prompts = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line.startswith('[') or not line.endswith(']'):
                        continue
                    
                    try:
                        content = line[1:-1]
                        parts = content.split(' ¥')
                        if len(parts) != 2:
                            self.logger.warning(f"Format tidak valid di baris {line_num}: Tanda '¥' tidak ditemukan atau lebih dari satu.")
                            continue
                            
                        workflow_part, prompt_text = parts
                        prompt_text = prompt_text.rstrip('¥').strip()
                        
                        workflow_pattern = r'∆\{([^}]+)\}•(\d+)∆'
                        matches = re.findall(workflow_pattern, workflow_part)
                        
                        if not matches:
                            self.logger.warning(f"Format workflow tidak ditemukan di baris {line_num}: {line}")
                            continue

                        ratios = [{'type': name.strip(), 'count': int(count)} for name, count in matches]
                        
                        prompts.append({
                            'text': prompt_text,
                            'ratios': ratios,
                            'line_num': line_num
                        })

                    except Exception as e:
                        self.logger.error(f"Error mem-parsing baris {line_num}: {line} | Error: {e}")
        except FileNotFoundError:
            self.logger.error(f"File prompt tidak ditemukan: {filepath}")
        return prompts

    def load_workflow(self, workflow_type):
        """Memuat file workflow JSON (Sudah dinamis)"""
        workflow_path = self.workflows.get(workflow_type)
        if not workflow_path:
            self.logger.error(f"Nama workflow '{workflow_type}' tidak ditemukan. Pastikan file '{workflow_type}.json' ada di folder 'workflows'.")
            raise FileNotFoundError(f"Workflow {workflow_type} tidak ditemukan")
        with open(workflow_path, 'r') as f:
            return json.load(f)

    def update_workflow_prompt(self, workflow, prompt_text):
        """Memperbarui prompt text dan seed di dalam workflow"""
        for node_data in workflow.values():
            if isinstance(node_data, dict):
                if node_data.get('class_type') == 'CLIPTextEncode' and 'inputs' in node_data and 'text' in node_data['inputs']:
                    node_data['inputs']['text'] = prompt_text
                if node_data.get('class_type') == 'KSampler' and 'inputs' in node_data:
                    node_data['inputs']['seed'] = random.randint(1, 10**15)
        return workflow

    def queue_prompt(self, workflow):
        """Mengirim pekerjaan ke antrian ComfyUI"""
        p = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        try:
            resp = requests.post(f"http://{self.server_address}/prompt", data=data)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self.logger.error(f"Gagal mengirim prompt ke ComfyUI: {e}")
            return None

    def get_history(self, prompt_id):
        """Mendapatkan riwayat dari ComfyUI"""
        try:
            resp = requests.get(f"http://{self.server_address}/history/{prompt_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def wait_for_completion(self, prompt_id, timeout=600):
        """Menunggu pekerjaan selesai"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if history and prompt_id in history:
                return True
            time.sleep(2)
        return False

    def process_prompts(self, prompt_file):
        """Memproses semua prompt dari file"""
        prompts = self.parse_prompt_file(prompt_file)
        if not prompts:
            self.logger.info("Tidak ada prompt valid untuk diproses.")
            return

        total_generations = sum(sum(r['count'] for r in p['ratios']) for p in prompts)
        self.logger.info(f"Memulai proses {len(prompts)} baris prompt dengan total {total_generations} gambar.")

        completed, failed = 0, 0
        for prompt_idx, prompt_data in enumerate(prompts):
            prompt_text = prompt_data['text']
            self.logger.info(f"\nMemproses baris {prompt_idx + 1}/{len(prompts)}: {prompt_text[:70]}...")
            
            for ratio_data in prompt_data['ratios']:
                ratio_type = ratio_data['type']
                count = ratio_data['count']
                for i in range(count):
                    try:
                        workflow = self.load_workflow(ratio_type)
                        workflow = self.update_workflow_prompt(workflow, prompt_text)
                        
                        result = self.queue_prompt(workflow)
                        if result and 'prompt_id' in result:
                            prompt_id = result['prompt_id']
                            self.logger.info(f" - Generating ({ratio_type}) {i+1}/{count} (ID: {prompt_id})")
                            
                            if self.wait_for_completion(prompt_id):
                                completed += 1
                                self.logger.info(f"   Selesai: {ratio_type} {i+1}/{count}")
                            else:
                                failed += 1
                                self.logger.error(f"   X Timeout: {ratio_type} {i+1}/{count}")
                        else:
                            failed += 1
                            self.logger.error(f"   X Gagal antri: {ratio_type} {i+1}/{count}")
                        time.sleep(2) 
                    
                    except Exception as e:
                        failed += 1
                        self.logger.error(f"   X Error Kritis pada ({ratio_type}): {e}")

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"SELESAI! Total: {total_generations}, Berhasil: {completed}, Gagal: {failed}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python comfyui_batch_processor_v2.py <prompt_file.txt>")
        sys.exit(1)
            
    prompt_file = sys.argv[1]
    if not os.path.exists(prompt_file):
        print(f"File tidak ditemukan: {prompt_file}")
        sys.exit(1)
            
    processor = ComfyUIBatchProcessorV2()
    processor.process_prompts(prompt_file)

                                                
