"use client";

import React, { useState, useRef } from 'react';
import { Settings, Brain, HelpCircle, UploadCloud, Target, X, Loader2, CheckCircle2, PlusCircle, Trash2 } from 'lucide-react';
import ReactCrop from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';

export default function ZeroShotCountingApp() {
  // --- STATES GIAO DIỆN & THÔNG SỐ ---
  const [cosineThreshold, setCosineThreshold] = useState(0.85);
  const [nmsThreshold, setNmsThreshold] = useState(0.25);
  
  // --- STATES XỬ LÝ ẢNH & VẼ BOX ---
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [crop, setCrop] = useState();
  const imageRef = useRef(null);
  
  // --- STATES LƯU NHIỀU MẪU (FEW-SHOT) ---
  const [realBoxes, setRealBoxes] = useState([]); // Tọa độ thực gửi cho AI
  const [uiBoxes, setUiBoxes] = useState([]); // Tọa độ % để vẽ lên Web
  
  // --- STATES KẾT QUẢ API ---
  const [isCounting, setIsCounting] = useState(false);
  const [resultData, setResultData] = useState(null); 

  const handleImageSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = () => setImageSrc(reader.result);
      reader.readAsDataURL(file);
      resetAllData();
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleImageSelect(e.dataTransfer.files[0]);
    }
  };

  const resetAllData = () => {
    setCrop(undefined);
    setResultData(null);
    setRealBoxes([]);
    setUiBoxes([]);
  };

  const resetUpload = () => {
    setImageSrc(null);
    setSelectedFile(null);
    resetAllData();
  };

  // --- HÀM LƯU MẪU (THÊM VÀO MẢNG) ---
  const handleSaveExemplar = () => {
    if (!imageRef.current || !crop || !crop.width || !crop.height) return;
    if (realBoxes.length >= 3) {
      alert("Bạn chỉ được chọn tối đa 3 mẫu để đảm bảo hiệu suất!");
      return;
    }

    // 1. Tính tọa độ thực (Gửi API)
    const scaleX = imageRef.current.naturalWidth / imageRef.current.width;
    const scaleY = imageRef.current.naturalHeight / imageRef.current.height;
    
    const rBbox = {
      x: Math.round(crop.x * scaleX),
      y: Math.round(crop.y * scaleY),
      w: Math.round(crop.width * scaleX),
      h: Math.round(crop.height * scaleY)
    };

    // 2. Tính tọa độ % (Để vẽ lưu lại trên UI không bị lệch khi resize)
    const pctBbox = {
      left: (crop.x / imageRef.current.width) * 100,
      top: (crop.y / imageRef.current.height) * 100,
      width: (crop.width / imageRef.current.width) * 100,
      height: (crop.height / imageRef.current.height) * 100,
    };

    setRealBoxes([...realBoxes, rBbox]);
    setUiBoxes([...uiBoxes, pctBbox]);
    setCrop(undefined); // Xóa khung vẽ hiện tại để vẽ cái mới
    setResultData(null);
  };

  const handleClearExemplars = () => {
    setRealBoxes([]);
    setUiBoxes([]);
    setCrop(undefined);
    setResultData(null);
  };

  // --- HÀM GỌI API ĐẾM ---
  const handleCountObjects = async () => {
    // Nếu người dùng đang vẽ dở 1 ô mà chưa bấm lưu, tự động lưu nó luôn
    let finalBoxes = [...realBoxes];
    if (crop && crop.width && finalBoxes.length < 3) {
      const scaleX = imageRef.current.naturalWidth / imageRef.current.width;
      const scaleY = imageRef.current.naturalHeight / imageRef.current.height;
      finalBoxes.push({
        x: Math.round(crop.x * scaleX),
        y: Math.round(crop.y * scaleY),
        w: Math.round(crop.width * scaleX),
        h: Math.round(crop.height * scaleY)
      });
    }

    if (finalBoxes.length === 0) {
      alert("Vui lòng khoanh vùng ít nhất 1 vật thể mẫu!");
      return;
    }

    setIsCounting(true);
    setResultData(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    // CHÚ Ý: Backend mới dùng biến 'boxes_str' (mảng JSON)
    formData.append("boxes_str", JSON.stringify(finalBoxes)); 
    formData.append("cosine_threshold", cosineThreshold);
    formData.append("nms_threshold", nmsThreshold);

    try {
      // ⚠️ ĐIỀN LINK HUGGING FACE CỦA BẠN VÀO ĐÂY
      const API_URL = "https://quan030106-zsc-by-mlp-and-vectorsearch.hf.space/predict/"; 
      
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Mã lỗi: ${response.status}. Chi tiết: ${errorText}`);
      }

      const data = await response.json();
      setResultData(data); 

    } catch (err) {
      console.error("Lỗi API:", err);
      alert(`Lỗi AI: ${err.message}`);
    } finally {
      setIsCounting(false);
      setCrop(undefined); // Xóa khung nháp đi
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans relative overflow-hidden flex flex-col md:flex-row p-4 md:p-8 gap-6">
      
      {/* Background */}
      <div className="absolute top-[-15%] left-[-10%] w-[500px] h-[500px] bg-cyan-600/30 rounded-full mix-blend-screen filter blur-[120px] pointer-events-none"></div>
      <div className="absolute top-[20%] right-[-10%] w-[600px] h-[600px] bg-purple-700/30 rounded-full mix-blend-screen filter blur-[150px] pointer-events-none"></div>
      <div className="absolute bottom-[-20%] left-[20%] w-[500px] h-[500px] bg-pink-600/30 rounded-full mix-blend-screen filter blur-[120px] pointer-events-none"></div>

      {/* SIDEBAR */}
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
            <input type="range" min="0.10" max="0.99" step="0.01" value={cosineThreshold} onChange={(e) => setCosineThreshold(parseFloat(e.target.value))} className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-pink-500"/>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium text-slate-300">Ngưỡng NMS</label>
              <span className="text-lg font-bold text-pink-400">{nmsThreshold.toFixed(2)}</span>
            </div>
            <input type="range" min="0.05" max="0.60" step="0.01" value={nmsThreshold} onChange={(e) => setNmsThreshold(parseFloat(e.target.value))} className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-pink-500"/>
          </div>
        </div>

        {resultData && (
          <div className="bg-gradient-to-br from-green-500/20 to-emerald-700/20 border border-green-400/30 rounded-2xl p-5 shadow-[0_0_15px_rgba(16,185,129,0.2)] animate-in fade-in">
            <h3 className="text-green-400 font-bold flex items-center gap-2 mb-2"><CheckCircle2 className="w-5 h-5"/> Kết quả đếm</h3>
            <div className="text-4xl font-black text-white mb-2">{resultData.object_count} <span className="text-lg font-medium text-slate-300">vật thể</span></div>
            <p className="text-sm text-slate-400">Thời gian xử lý: <strong className="text-white">{resultData.latency_sec}s</strong></p>
          </div>
        )}

        <div className="mt-auto bg-slate-900/60 border border-white/5 rounded-2xl p-5 shadow-inner">
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-5 h-5 text-purple-400" />
            <h3 className="font-semibold text-slate-100">Few-Shot AI Engine</h3>
          </div>
          <ul className="text-sm text-slate-400 space-y-2 mb-3">
            <li>• HOG+HSV+LBP (2068-dim)</li>
            <li>• PCA Whitening (128-dim)</li>
            <li>• Few-Shot Mean Pooling</li>
          </ul>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="w-full md:w-2/3 lg:w-3/4 bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl flex flex-col items-center z-10 relative overflow-y-auto">
        <div className="text-center mb-4 w-full">
          <h1 className="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-pink-500 via-purple-400 to-cyan-400">
            Few-Shot Object Counting
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
            
            {/* Toolbar trên ảnh */}
            <div className="flex items-center justify-between w-full max-w-4xl bg-slate-900/50 p-3 rounded-2xl border border-slate-700">
              <div className="flex items-center gap-3">
                <span className="text-cyan-300 font-medium text-sm flex items-center gap-1">
                  <Target className="w-4 h-4" /> Mẫu: {realBoxes.length}/3
                </span>
                
                <button 
                  onClick={handleSaveExemplar} 
                  disabled={!crop || !crop.width || realBoxes.length >= 3}
                  className="px-3 py-1.5 bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 rounded-lg text-sm font-semibold hover:bg-cyan-500 hover:text-white transition-all disabled:opacity-30 flex items-center gap-1"
                >
                  <PlusCircle className="w-4 h-4" /> Lưu Mẫu Này
                </button>

                {realBoxes.length > 0 && (
                  <button onClick={handleClearExemplars} className="px-3 py-1.5 text-rose-400 hover:text-rose-300 hover:bg-rose-900/30 rounded-lg text-sm font-medium transition-all flex items-center gap-1">
                    <Trash2 className="w-4 h-4" /> Xóa Mẫu
                  </button>
                )}
              </div>

              <button onClick={resetUpload} className="text-slate-400 hover:text-white text-sm font-semibold transition-colors flex items-center gap-1">
                <X className="w-4 h-4" /> Chọn ảnh khác
              </button>
            </div>

            {/* VÙNG CHỨA ẢNH */}
            <div className="relative border border-slate-700 rounded-xl bg-black/50 p-2 max-w-full overflow-hidden flex justify-center shadow-2xl">
              <ReactCrop 
                crop={crop} 
                onChange={c => { setCrop(c); if(resultData) setResultData(null); }} 
                disabled={realBoxes.length >= 3 && !resultData}
              >
                <div className="relative inline-block">
                  <img ref={imageRef} src={imageSrc} alt="Upload" className="max-h-[55vh] w-auto object-contain block" />
                  
                  {/* Vẽ các ô mẫu đã lưu (Màu Xanh Ngọc) */}
                  {uiBoxes.map((b, idx) => (
                    <div key={`ui-${idx}`} className="absolute border-2 border-dashed border-cyan-400 bg-cyan-400/20 pointer-events-none"
                         style={{ left: `${b.left}%`, top: `${b.top}%`, width: `${b.width}%`, height: `${b.height}%` }} />
                  ))}

                  {/* Vẽ kết quả AI (Màu Neon Pink) */}
                  {resultData && imageRef.current && resultData.bounding_boxes.map((box, idx) => (
                    <div key={`res-${idx}`} className="absolute border-2 border-[#ff0055] bg-[#ff0055]/20 shadow-[0_0_8px_rgba(255,0,85,0.8)] pointer-events-none"
                         style={{ left: `${(box.x / imageRef.current.naturalWidth) * 100}%`, top: `${(box.y / imageRef.current.naturalHeight) * 100}%`, width: `${(box.w / imageRef.current.naturalWidth) * 100}%`, height: `${(box.h / imageRef.current.naturalHeight) * 100}%` }} />
                  ))}
                </div>
              </ReactCrop>
            </div>

            {/* NÚT BẤM KÍCH HOẠT */}
            <button 
              onClick={handleCountObjects} 
              disabled={isCounting || (realBoxes.length === 0 && (!crop || !crop.width))} 
              className={`mt-2 px-10 py-4 rounded-full font-bold text-lg shadow-[0_0_20px_rgba(236,72,153,0.3)] transition-all flex items-center gap-3 
                ${(realBoxes.length === 0 && (!crop || !crop.width)) ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700' : 'bg-gradient-to-r from-pink-600 to-purple-600 text-white hover:scale-105 border border-pink-500/50'}`}
            >
              {isCounting ? (
                <><Loader2 className="w-6 h-6 animate-spin" /> Phân tích...</>
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
