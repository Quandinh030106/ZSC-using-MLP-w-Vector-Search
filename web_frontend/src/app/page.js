"use client";

import React, { useState, useRef } from 'react';
import { Settings, Brain, HelpCircle, UploadCloud, Target, X, Loader2, CheckCircle2 } from 'lucide-react';
import ReactCrop from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';

export default function ZeroShotCountingApp() {
  // --- STATES GIAO DIỆN & THÔNG SỐ ---
  const [cosineThreshold, setCosineThreshold] = useState(0.90);
  const [nmsThreshold, setNmsThreshold] = useState(0.20);
  
  // --- STATES XỬ LÝ ẢNH & VẼ BOX ---
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [crop, setCrop] = useState();
  const imageRef = useRef(null);
  
  // --- STATES KẾT QUẢ API ---
  const [isCounting, setIsCounting] = useState(false);
  const [resultData, setResultData] = useState(null); 

  const handleImageSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = () => setImageSrc(reader.result);
      reader.readAsDataURL(file);
      setResultData(null); 
      setCrop(undefined);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleImageSelect(e.dataTransfer.files[0]);
    }
  };

  const resetUpload = () => {
    setImageSrc(null);
    setSelectedFile(null);
    setCrop(undefined);
    setResultData(null);
  };

  const handleCountObjects = async () => {
    if (!imageRef.current || !crop || !crop.width || !crop.height) {
      alert("Vui lòng khoanh vùng một vật thể mẫu trên ảnh!");
      return;
    }

    setIsCounting(true);
    setResultData(null);

    // Tính toán tọa độ ảnh thật để gửi cho AI
    const scaleX = imageRef.current.naturalWidth / imageRef.current.width;
    const scaleY = imageRef.current.naturalHeight / imageRef.current.height;
    
    const realBbox = {
      x: Math.round(crop.x * scaleX),
      y: Math.round(crop.y * scaleY),
      w: Math.round(crop.width * scaleX),
      h: Math.round(crop.height * scaleY)
    };

    console.log("Đang gửi dữ liệu lên Backend:", { file: selectedFile.name, bbox: realBbox });

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("bbox_str", JSON.stringify(realBbox));
    formData.append("cosine_threshold", cosineThreshold);
    formData.append("nms_threshold", nmsThreshold);
    
    const dynamicWindowSize = Math.max(realBbox.w, realBbox.h); 
    const dynamicStepSize = Math.max(16, Math.floor(dynamicWindowSize / 2)); // Trượt 50% kích thước ô

    formData.append("window_size", dynamicWindowSize);
    formData.append("step_size", dynamicStepSize);

    try {
      // LINK HUGGING FACE CỦA BẠN 
      const API_URL = "https://quan030106-zsc-by-mlp-and-vectorsearch.hf.space/predict/"; 
      
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      // BỘ DÒ LỖI CHI TIẾT
      if (!response.ok) {
        const errorText = await response.text();
        console.error("Lỗi Backend trả về:", errorText);
        throw new Error(`Mã lỗi: ${response.status}. Chi tiết: ${errorText}`);
      }

      const data = await response.json();
      console.log("Nhận kết quả thành công:", data);
      setResultData(data); 

    } catch (err) {
      console.error("Lỗi quá trình gọi API:", err);
      alert(`Không thể lấy kết quả AI.\nLý do: ${err.message}\n\nHãy mở F12 (Console) trên trình duyệt hoặc xem Logs trên Hugging Face để biết thêm chi tiết.`);
    } finally {
      setIsCounting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans relative overflow-hidden flex flex-col md:flex-row p-4 md:p-8 gap-6">
      
      {/* Background Gradients */}
      <div className="absolute top-[-15%] left-[-10%] w-[500px] h-[500px] bg-cyan-600/30 rounded-full mix-blend-screen filter blur-[120px] pointer-events-none"></div>
      <div className="absolute top-[20%] right-[-10%] w-[600px] h-[600px] bg-purple-700/30 rounded-full mix-blend-screen filter blur-[150px] pointer-events-none"></div>
      <div className="absolute bottom-[-20%] left-[20%] w-[500px] h-[500px] bg-pink-600/30 rounded-full mix-blend-screen filter blur-[120px] pointer-events-none"></div>

      {/* SIDEBAR BẢNG ĐIỀU KHIỂN */}
      <aside className="w-full md:w-1/3 lg:w-1/4 bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6 shadow-2xl flex flex-col gap-8 z-10">
        <div className="flex items-center gap-3 border-b border-white/10 pb-4">
          <Settings className="w-7 h-7 text-cyan-400 animate-[spin_4s_linear_infinite]" />
          <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-400">Bảng Điều Khiển</h2>
        </div>

        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium text-slate-300">Ngưỡng Cosine</label>
              <span className="text-lg font-bold text-pink-400">{cosineThreshold.toFixed(2)}</span>
            </div>
            <input type="range" min="0.70" max="0.99" step="0.01" value={cosineThreshold} onChange={(e) => setCosineThreshold(parseFloat(e.target.value))} className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-pink-500"/>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium text-slate-300">Ngưỡng NMS</label>
              <span className="text-lg font-bold text-pink-400">{nmsThreshold.toFixed(2)}</span>
            </div>
            <input type="range" min="0.05" max="0.50" step="0.01" value={nmsThreshold} onChange={(e) => setNmsThreshold(parseFloat(e.target.value))} className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-pink-500"/>
          </div>
        </div>

        {/* Khối hiển thị Kết Quả Nổi Bật (Chỉ hiện khi đếm xong) */}
        {resultData && (
          <div className="bg-gradient-to-br from-green-500/20 to-emerald-700/20 border border-green-400/30 rounded-2xl p-5 shadow-[0_0_15px_rgba(16,185,129,0.2)] animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h3 className="text-green-400 font-bold flex items-center gap-2 mb-2"><CheckCircle2 className="w-5 h-5"/> Kết quả phân tích</h3>
            <div className="text-4xl font-black text-white mb-2">{resultData.object_count} <span className="text-lg font-medium text-slate-300">vật thể</span></div>
            <p className="text-sm text-slate-400">Thời gian xử lý: <strong className="text-white">{resultData.latency_sec}s</strong></p>
          </div>
        )}

        <div className="mt-auto bg-slate-900/60 border border-white/5 rounded-2xl p-5 shadow-inner">
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-5 h-5 text-purple-400" />
            <h3 className="font-semibold text-slate-100">Kiến Trúc AI</h3>
          </div>
          <ul className="text-sm text-slate-400 space-y-2 mb-3">
            <li>Feature: HOG+HSV+LBP (2068-dim)</li>
            <li>Model: Siamese MLP (11M params)</li>
            <li>Search: FAISS (IndexFlatIP)</li>
          </ul>
          <div className="p-2 bg-amber-500/10 border border-amber-500/20 rounded-xl text-center"><p className="text-xs italic text-amber-400">* KHÔNG dùng CNN/Transformer *</p></div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="w-full md:w-2/3 lg:w-3/4 bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl flex flex-col items-center z-10 relative overflow-y-auto">
        <div className="text-center mb-6 w-full">
          <h1 className="text-4xl md:text-5xl font-extrabold mb-2 tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-pink-500 via-purple-400 to-cyan-400">
            Zero-Shot Object Counting
          </h1>
        </div>

        {!imageSrc ? (
          <div className="w-full max-w-2xl flex flex-col gap-3 mt-10">
            <div className={`relative group flex flex-col items-center justify-center w-full h-80 rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer bg-slate-900/40 ${isDragging ? 'border-cyan-400 bg-cyan-900/20' : 'border-slate-600 hover:border-pink-500 hover:bg-slate-800/60'}`} onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={onDrop}>
              <UploadCloud className={`w-16 h-16 mb-4 ${isDragging ? 'text-cyan-400' : 'text-slate-400 group-hover:text-pink-400'}`} />
              <p className="mb-2 text-lg font-bold text-slate-200">Kéo và thả ảnh vào đây</p>
              <label className="mt-4 px-6 py-2.5 rounded-full bg-slate-800 border border-slate-600 font-semibold hover:bg-slate-700 hover:border-pink-500 hover:text-white transition-all cursor-pointer">
                Chọn file từ máy
                <input type="file" className="hidden" accept="image/*" onChange={(e) => handleImageSelect(e.target.files[0])} />
              </label>
            </div>
          </div>
        ) : (
          <div className="w-full flex flex-col items-center gap-4">
            <div className="flex items-center justify-between w-full max-w-4xl">
              <p className="text-cyan-300 font-medium flex items-center gap-2">
                <Target className="w-5 h-5" /> Kéo chuột để khoanh vùng mẫu
              </p>
              <button onClick={resetUpload} className="text-slate-400 hover:text-pink-400 flex items-center gap-1 text-sm font-semibold transition-colors">
                <X className="w-4 h-4" /> Hủy / Trở lại
              </button>
            </div>

            {/* KHU VỰC CHỨA ẢNH & BOXES */}
            <div className="relative border border-slate-700 rounded-xl bg-black/50 p-2 max-w-full overflow-hidden flex justify-center shadow-2xl">
              <ReactCrop 
                crop={crop} 
                onChange={c => {
                  setCrop(c);
                  // Nếu người dùng vẽ lại ô vuông khác, lập tức xóa các kết quả cũ đi
                  if (resultData) setResultData(null); 
                }} 
              >
                <div className="relative inline-block">
                  <img ref={imageRef} src={imageSrc} alt="Upload" className="max-h-[60vh] w-auto object-contain block" />
                  
                  {/* VẼ KẾT QUẢ BOUNDING BOXES TỪ API */}
                  {resultData && imageRef.current && resultData.bounding_boxes.map((box, idx) => (
                    <div 
                      key={idx}
                      className="absolute border-2 border-[#ff0055] bg-[#ff0055]/20 shadow-[0_0_8px_rgba(255,0,85,0.8)]"
                      style={{
                        left: `${(box.x / imageRef.current.naturalWidth) * 100}%`,
                        top: `${(box.y / imageRef.current.naturalHeight) * 100}%`,
                        width: `${(box.w / imageRef.current.naturalWidth) * 100}%`,
                        height: `${(box.h / imageRef.current.naturalHeight) * 100}%`,
                      }}
                    />
                  ))}
                </div>
              </ReactCrop>
            </div>

            {/* NÚT BẤM KÍCH HOẠT AI (Luôn hiện để đếm lại) */}
            <button 
              onClick={handleCountObjects} 
              disabled={isCounting || !crop || !crop.width} 
              className={`mt-4 px-10 py-4 rounded-full font-bold text-lg shadow-[0_0_20px_rgba(236,72,153,0.3)] transition-all flex items-center gap-3 
                ${(!crop || !crop.width) ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700' 
                : 'bg-gradient-to-r from-pink-600 to-purple-600 text-white hover:scale-105 border border-pink-500/50'}`}
            >
              {isCounting ? (
                <><Loader2 className="w-6 h-6 animate-spin" /> Đang chạy AI xử lý...</>
              ) : resultData ? (
                <><Target className="w-6 h-6" /> Áp Dụng Ngưỡng Mới</>
              ) : (
                <><Target className="w-6 h-6" /> Đếm Vật Thể</>
              )}
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
