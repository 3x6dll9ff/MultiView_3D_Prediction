import { useState } from 'react'
import { Scene } from '../App'

interface CustomGeneratorProps {
  vaeAvailable: boolean
}

export default function CustomGenerator({ vaeAvailable }: CustomGeneratorProps) {
  const [files, setFiles] = useState<{ top: File | null; bottom: File | null; side: File | null }>({
    top: null,
    bottom: null,
    side: null,
  })
  
  const [previews, setPreviews] = useState<{ top: string; bottom: string; side: string }>({
    top: '',
    bottom: '',
    side: '',
  })

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ cnn_mesh: any; vae_mesh: any } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (view: 'top' | 'bottom' | 'side') => (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setFiles(prev => ({ ...prev, [view]: file }))
      
      const reader = new FileReader()
      reader.onload = (ev) => {
        setPreviews(prev => ({ ...prev, [view]: ev.target?.result as string }))
      }
      reader.readAsDataURL(file)
    }
  }

  const handleGenerate = async () => {
    if (!files.top || !files.bottom || !files.side) {
      setError('Пожалуйста, загрузите все 3 фотографии (Top, Bottom, Side).')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('top', files.top)
    formData.append('bottom', files.bottom)
    formData.append('side', files.side)

    try {
      // Испольузем относительный путь с учетом проксирования vite
      const backendUrl = import.meta.env.VITE_API_BASE_URL || ''
      const response = await fetch(`${backendUrl}/api/generate-custom`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Ошибка сервера: ${response.statusText}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'Произошла ошибка при генерации.')
    } finally {
      setLoading(false)
    }
  }

  const allUploaded = files.top && files.bottom && files.side

  return (
    <div style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto', width: '100%' }}>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
          Custom 3D Generator
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', maxWidth: '600px' }}>
          Upload your own top, bottom, and side projections of a cell to generate its 3D morphology using both our CNN + Refiner and Conditional VAE models.
        </p>
      </header>

      <div className="upload-grid">
        {(['top', 'bottom', 'side'] as const).map(view => (
          <div className="upload-box" key={view}>
            <input 
              type="file" 
              accept="image/*" 
              className="upload-input" 
              onChange={handleFileChange(view)}
            />
            <div className="upload-label">{view.toUpperCase()} VIEW</div>
            
            {previews[view] ? (
              <img src={previews[view]} alt={view} className="upload-preview" />
            ) : (
              <div className="upload-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                  <circle cx="8.5" cy="8.5" r="1.5"/>
                  <polyline points="21 15 16 10 5 21"/>
                </svg>
              </div>
            )}
          </div>
        ))}
      </div>

      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', padding: '12px', borderRadius: '4px', marginBottom: '24px', fontSize: '13px', border: '1px solid rgba(239,68,68,0.2)' }}>
          {error}
        </div>
      )}

      <button 
        className="upload-btn" 
        onClick={handleGenerate} 
        disabled={!allUploaded || loading}
        style={{ marginBottom: '32px', maxWidth: '200px' }}
      >
        {loading ? 'Generating...' : 'Generate 3D'}
        {!loading && (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        )}
      </button>

      {result && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: '32px' }}>
          <h2 style={{ fontSize: '18px', fontWeight: 500, marginBottom: '24px' }}>Generated Models</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px' }}>
            <div style={{ background: '#0a0a12', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', overflow: 'hidden', height: '400px', position: 'relative' }}>
              <Scene 
                meshData={result.cnn_mesh} 
                color="#4fffff" 
                label="CNN + Refiner Model" 
                syncedControls={false}
                fitCamera
              />
            </div>
            
            {vaeAvailable && result.vae_mesh ? (
              <div style={{ background: '#0a0a12', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', overflow: 'hidden', height: '400px', position: 'relative' }}>
                <Scene 
                  meshData={result.vae_mesh} 
                  color="#a0c4ff" 
                  label="Conditional VAE Model" 
                  syncedControls={false}
                  fitCamera
                />
              </div>
            ) : (
              <div style={{ background: '#0a0a12', border: '1px dashed var(--border)', borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '400px', color: 'var(--text-muted)' }}>
                VAE model not available or did not generate
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
