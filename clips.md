# 🎬 Clips Service - Cloud Run Implementation Plan

## 📋 Project Overview

### Purpose
Transform the standalone ClippedAI script into a scalable, GPU-accelerated Cloud Run service that processes video clips with AI-powered captioning, resizing, and viral title generation.

### Key Changes from Current Implementation
- Convert from CLI script to FastAPI service
- Integrate with GCS Pub/Sub messaging
- Use Firestore for task state tracking
- Work with separated video/audio files
- Generate word-level timestamps on-demand from segment-level transcriptions
- Task-isolated processing to prevent cross-contamination
- GPU-accelerated video processing (NVENC)
- RESTful configuration for caption styling

---

## 🏗️ Architecture Design

### High-Level Flow
```
Pub/Sub Message → FastAPI Endpoint → Firestore Task Doc → Download Assets → 
stable-ts (word-level) → ClipsAI Processing → Caption Rendering → 
Upload Results → Update Firestore → Cleanup
```

### Component Breakdown

#### 1. **Message Handler** (`handlers/pubsub_handler.py`)
- Receive Pub/Sub messages
- Validate message payload
- Extract `task_id` and configuration
- Route to appropriate processor

#### 2. **Task Manager** (`services/task_manager.py`)
- Create task-specific working directories
- Track task lifecycle in Firestore
- Handle task cleanup
- Manage concurrent tasks

#### 3. **Asset Retriever** (`services/asset_retriever.py`)
- Download video file from GCS
- Download audio file from GCS
- Fetch transcription JSON from Firestore
- Merge video + audio into single file

#### 4. **Transcription Processor** (`services/transcription_processor.py`)
- Load stable-ts model from GCS bucket
- Convert segment-level → word-level timestamps
- Format for ClipsAI compatibility
- Cache word-level data (in-memory only)

#### 5. **Clip Generator** (`services/clip_generator.py`)
- Adapted from current `main.py`
- Find engaging clips using ClipsAI
- Score clips by engagement metrics
- Generate viral titles via Groq API

#### 6. **Caption Renderer** (`services/caption_renderer.py`)
- Generate ASS subtitle files
- Support configurable positioning, fonts, colors
- Apply animations (optional)
- Burn subtitles using FFmpeg + NVENC

#### 7. **Video Processor** (`services/video_processor.py`)
- Trim clips using FFmpeg + NVENC
- Resize to 9:16 aspect ratio
- Apply GPU-accelerated encoding
- Optimize for social media platforms

#### 8. **Uploader** (`services/uploader.py`)
- Upload processed clips to GCS
- Update Firestore with result URLs
- Generate thumbnail previews

---

## 📁 Project Structure

```
clips-service/
├── app/
│   ├── main.py                          # FastAPI application
│   ├── config.py                        # Configuration management
│   ├── schemas.py                       # Pydantic models
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── clips.py                # Clip generation endpoints
│   │   │   └── health.py               # Health check endpoints
│   │
│   ├── handlers/
│   │   ├── __init__.py
│   │   └── pubsub_handler.py           # Pub/Sub message handler
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── task_manager.py             # Task lifecycle management
│   │   ├── asset_retriever.py          # GCS/Firestore asset fetching
│   │   ├── transcription_processor.py   # stable-ts integration
│   │   ├── clip_generator.py           # ClipsAI clip detection
│   │   ├── caption_renderer.py         # Subtitle generation
│   │   ├── video_processor.py          # FFmpeg video processing
│   │   ├── uploader.py                 # Result upload to GCS
│   │   └── groq_client.py              # Viral title generation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip.py                     # Clip data model
│   │   ├── transcription.py            # Transcription data model
│   │   └── task.py                     # Task data model
│   │
│   └── utils/
│       ├── __init__.py
│       ├── ffmpeg_utils.py             # FFmpeg helper functions
│       ├── gcs_utils.py                # GCS helper functions
│       ├── firestore_utils.py          # Firestore helper functions
│       └── validation.py               # Input validation
│
├── tmp/                                 # Task-specific temp folders (gitignored)
│   └── {task_id}/                      # One folder per task
│       ├── input/
│       │   ├── video.mp4
│       │   ├── audio.mp3
│       │   └── transcription.json
│       ├── processing/
│       │   └── merged_video.mp4
│       └── output/
│           ├── clip_1.mp4
│           ├── clip_2.mp4
│           └── metadata.json
│
├── models/                              # Cached AI models
│   └── stable-ts/                      # Downloaded from GCS on startup
│
├── fonts/                               # Font files for captions
│   ├── Montserrat-ExtraBold.otf
│   ├── Arial-Black.ttf
│   └── ...
│
├── Dockerfile                           # GPU-enabled container
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 Technical Requirements

### Container Configuration

#### Dockerfile Requirements
```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install FFmpeg with NVENC support
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3.10 \
    python3-pip \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download stable-ts model from GCS on startup
COPY download_models.sh /app/
RUN chmod +x /app/download_models.sh

WORKDIR /app
COPY . .

# Ensure GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Python Dependencies
```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.6.0
python-dotenv==1.0.0
google-cloud-storage==2.14.0
google-cloud-firestore==2.14.0
google-cloud-pubsub==2.19.0
clipsai==0.2.1
stable-ts==2.17.0
requests==2.31.0
python-magic==0.4.27
aiofiles==23.2.1
```

### GPU Configuration
- **FFmpeg**: Compile with `--enable-nvenc --enable-cuda`
- **NVENC Codecs**: h264_nvenc, hevc_nvenc
- **stable-ts**: CUDA-enabled PyTorch

### GCS Bucket Structure
```
gs://{your-bucket}/
├── models/
│   └── stable-ts/
│       └── medium.pt
├── input/
│   └── {task_id}/
│       ├── video.mp4
│       └── audio.mp3
└── output/
    └── {task_id}/
        ├── clip_1.mp4
        ├── clip_2.mp4
        └── metadata.json
```

### Firestore Schema
```javascript
// Collection: tasks
{
  task_id: "uuid-v4",
  status: "pending | processing | completed | failed",
  created_at: Timestamp,
  updated_at: Timestamp,
  
  // Input references
  input: {
    video_url: "gs://bucket/input/{task_id}/video.mp4",
    audio_url: "gs://bucket/input/{task_id}/audio.mp3",
    transcription: {
      segments: [
        {
          start: 0.0,
          end: 5.2,
          text: "Hello world"
        }
      ]
    }
  },
  
  // Configuration
  config: {
    num_clips: 2,
    min_duration: 45,
    max_duration: 120,
    subtitle_config: {
      enabled: true,
      position: "top-center",
      font: {
        family: "Montserrat-ExtraBold",
        size: 80,
        color: "#FFFFFF"
      }
    }
  },
  
  // Output results
  output: {
    clips: [
      {
        clip_id: 1,
        url: "gs://bucket/output/{task_id}/clip_1.mp4",
        title: "Viral Title 🎉",
        duration: 60.5,
        start_time: 120.0,
        end_time: 180.5,
        engagement_score: 0.85
      }
    ]
  },
  
  // Error tracking
  error: {
    message: null,
    stack_trace: null
  }
}
```

---

## 📡 API Specification

### 1. Pub/Sub Endpoint
```
POST /api/v1/clips/process
Content-Type: application/json

{
  "message": {
    "data": "base64_encoded_payload",
    "attributes": {
      "task_id": "uuid-v4"
    }
  }
}

Decoded payload:
{
  "task_id": "uuid-v4",
  "action": "generate_clips",
  "config": {
    "num_clips": 2,
    "subtitle_config": { ... }
  }
}
```

### 2. Direct HTTP Endpoint (Optional)
```
POST /api/v1/clips/generate
Content-Type: application/json

{
  "task_id": "uuid-v4",
  "video_url": "gs://bucket/input/{task_id}/video.mp4",
  "audio_url": "gs://bucket/input/{task_id}/audio.mp3",
  "transcription": {
    "segments": [ ... ]
  },
  "config": {
    "num_clips": 2,
    "min_duration": 45,
    "max_duration": 120,
    "subtitle_config": {
      "enabled": true,
      "position": "top-center",
      "font": {
        "family": "Montserrat-ExtraBold",
        "size": 80,
        "color": "#FFFFFF",
        "outline_size": 15,
        "outline_color": "#000000"
      },
      "accent_color": "#FFFF00",
      "max_chars_per_line": 25,
      "animation": {
        "type": "none",
        "duration": 0.3
      }
    }
  }
}

Response:
{
  "task_id": "uuid-v4",
  "status": "processing",
  "message": "Clip generation started"
}
```

### 3. Status Endpoint
```
GET /api/v1/clips/{task_id}/status

Response:
{
  "task_id": "uuid-v4",
  "status": "completed",
  "clips": [
    {
      "clip_id": 1,
      "url": "gs://bucket/output/{task_id}/clip_1.mp4",
      "title": "Viral Title 🎉",
      "duration": 60.5
    }
  ]
}
```

### 4. Health Check
```
GET /health

Response:
{
  "status": "healthy",
  "gpu_available": true,
  "models_loaded": true
}
```

---

## 🔄 Data Flow

### Phase 1: Initialization
```python
1. Receive Pub/Sub message → extract task_id
2. Create task directory: tmp/{task_id}/
3. Update Firestore: status = "processing"
4. Load task config from Firestore
```

### Phase 2: Asset Retrieval
```python
5. Download video from GCS → tmp/{task_id}/input/video.mp4
6. Download audio from GCS → tmp/{task_id}/input/audio.mp3
7. Fetch transcription JSON from Firestore
8. Merge video + audio using FFmpeg:
   ffmpeg -i video.mp4 -i audio.mp3 -c:v copy -c:a aac \
          -map 0:v:0 -map 1:a:0 merged_video.mp4
```

### Phase 3: Transcription Processing
```python
9. Load stable-ts model (cached from GCS bucket)
10. Generate word-level timestamps from segment-level:
    
    Input (segment-level):
    {
      "segments": [
        {"start": 0.0, "end": 5.2, "text": "Hello world"}
      ]
    }
    
    Output (word-level):
    [
      {"word": "Hello", "start_time": 0.0, "end_time": 2.5},
      {"word": "world", "start_time": 2.6, "end_time": 5.2}
    ]
    
11. Format for ClipsAI compatibility (matches current Transcription object)
```

### Phase 4: Clip Generation
```python
12. Use ClipsAI ClipFinder to identify engaging moments
13. Calculate engagement scores for each clip
14. Select top N clips based on config.num_clips
15. Generate viral titles using Groq API
16. Store clip metadata in memory
```

### Phase 5: Video Processing
```python
For each selected clip:
  17. Trim video using FFmpeg + NVENC:
      ffmpeg -hwaccel cuda -i merged_video.mp4 \
             -ss {start} -to {end} \
             -c:v h264_nvenc -preset fast \
             tmp/{task_id}/processing/clip_{n}.mp4
  
  18. Resize to 9:16 using ClipsAI or FFmpeg:
      ffmpeg -hwaccel cuda -i clip_{n}.mp4 \
             -vf "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" \
             -c:v h264_nvenc \
             tmp/{task_id}/processing/clip_{n}_resized.mp4
```

### Phase 6: Caption Rendering
```python
  19. Generate ASS subtitle file with user config:
      - Position from config.subtitle_config.position
      - Font from config.subtitle_config.font
      - Colors from config.subtitle_config.color + accent_color
  
  20. Burn subtitles using FFmpeg + NVENC:
      ffmpeg -hwaccel cuda -i clip_{n}_resized.mp4 \
             -vf "ass=subtitles.ass" \
             -c:v h264_nvenc -preset fast \
             tmp/{task_id}/output/clip_{n}.mp4
```

### Phase 7: Upload & Finalize
```python
21. Upload all clips to GCS: gs://bucket/output/{task_id}/
22. Update Firestore with clip URLs and metadata
23. Update task status: "completed"
24. Cleanup: Remove tmp/{task_id}/ directory
```

---

## 🎨 Caption Configuration Schema

### Pydantic Models

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class CaptionPosition(str, Enum):
    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    MIDDLE_LEFT = "middle-left"
    MIDDLE_CENTER = "middle-center"
    MIDDLE_RIGHT = "middle-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"

class AnimationType(str, Enum):
    NONE = "none"
    FADE = "fade"
    KARAOKE = "karaoke"
    SLIDE = "slide"

class FontConfig(BaseModel):
    family: str = Field(default="Montserrat-ExtraBold", description="Font family name")
    size: int = Field(default=80, ge=20, le=200, description="Font size in pixels")
    color: str = Field(default="#FFFFFF", regex=r"^#[0-9A-Fa-f]{6}$", description="Primary text color")
    outline_size: int = Field(default=15, ge=0, le=50, description="Outline thickness")
    outline_color: str = Field(default="#000000", regex=r"^#[0-9A-Fa-f]{6}$", description="Outline color")
    bold: bool = Field(default=True, description="Bold text")
    italic: bool = Field(default=False, description="Italic text")

class AnimationConfig(BaseModel):
    type: AnimationType = Field(default=AnimationType.NONE, description="Animation type")
    duration: float = Field(default=0.3, ge=0.1, le=2.0, description="Animation duration in seconds")

class SubtitleConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable subtitle generation")
    position: CaptionPosition = Field(default=CaptionPosition.TOP_CENTER, description="Caption position")
    font: FontConfig = Field(default_factory=FontConfig, description="Font configuration")
    accent_color: str = Field(default="#FFFF00", regex=r"^#[0-9A-Fa-f]{6}$", description="Accent color for numbers/currency")
    max_chars_per_line: int = Field(default=25, ge=10, le=50, description="Maximum characters per line")
    animation: AnimationConfig = Field(default_factory=AnimationConfig, description="Animation settings")
    vertical_margin: int = Field(default=120, ge=0, le=500, description="Margin from edge in pixels")

class ClipConfig(BaseModel):
    num_clips: int = Field(default=2, ge=1, le=10, description="Number of clips to generate")
    min_duration: int = Field(default=45, ge=15, le=180, description="Minimum clip duration in seconds")
    max_duration: int = Field(default=120, ge=30, le=300, description="Maximum clip duration in seconds")
    subtitle_config: SubtitleConfig = Field(default_factory=SubtitleConfig, description="Subtitle configuration")
    engagement_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum engagement score for clips 3+")

class TaskPayload(BaseModel):
    task_id: str
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    transcription: Optional[dict] = None
    config: ClipConfig = Field(default_factory=ClipConfig)
```

---

## 🔨 Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
**Goal**: Set up FastAPI service with Pub/Sub integration

**Tasks**:
- [ ] Set up FastAPI project structure
- [ ] Implement Pub/Sub message handler
- [ ] Create Firestore task manager
- [ ] Implement task-based directory isolation
- [ ] Add health check endpoints
- [ ] Set up logging and monitoring

**Deliverables**:
- Working FastAPI service
- Pub/Sub message reception
- Task lifecycle management
- Basic error handling

---

### Phase 2: Asset Management (Week 1-2)
**Goal**: Download and prepare video assets

**Tasks**:
- [ ] Implement GCS asset downloader
- [ ] Fetch transcription from Firestore
- [ ] Merge video + audio using FFmpeg
- [ ] Add retry logic for network failures
- [ ] Implement cleanup on task completion/failure

**Deliverables**:
- Asset retrieval service
- Video/audio merging
- Robust error handling

---

### Phase 3: Transcription Processing (Week 2)
**Goal**: Convert segment-level to word-level timestamps

**Tasks**:
- [ ] Download stable-ts model from GCS bucket on startup
- [ ] Implement segment → word-level conversion
- [ ] Format transcription for ClipsAI compatibility
- [ ] Add caching for processed transcriptions (in-memory)
- [ ] Optimize for GPU acceleration

**Deliverables**:
- stable-ts integration
- Word-level timestamp generation
- ClipsAI-compatible format

---

### Phase 4: Clip Detection (Week 2-3)
**Goal**: Find engaging clips using AI

**Tasks**:
- [ ] Integrate ClipsAI ClipFinder
- [ ] Implement engagement scoring algorithm
- [ ] Add clip selection logic
- [ ] Integrate Groq API for title generation
- [ ] Handle edge cases (no clips found, all clips too short/long)

**Deliverables**:
- Working clip detection
- Engagement scoring
- Viral title generation

---

### Phase 5: Video Processing (Week 3)
**Goal**: Trim and resize clips with GPU acceleration

**Tasks**:
- [ ] Implement FFmpeg wrapper with NVENC support
- [ ] Add video trimming functionality
- [ ] Implement 9:16 resize with ClipsAI
- [ ] Add timecode track stripping
- [ ] Optimize encoding settings for quality/speed

**Deliverables**:
- GPU-accelerated video processing
- 9:16 aspect ratio conversion
- Optimized encoding

---

### Phase 6: Caption System (Week 3-4)
**Goal**: Configurable caption rendering

**Tasks**:
- [ ] Create ASS subtitle generator
- [ ] Implement configurable positioning
- [ ] Add font/color customization
- [ ] Support dynamic accent colors (numbers/currency)
- [ ] Add animation support (fade, karaoke, slide)
- [ ] Burn subtitles using FFmpeg + NVENC

**Deliverables**:
- Flexible caption system
- Multiple style presets
- Animation support
- GPU-accelerated rendering

---

### Phase 7: Upload & Finalization (Week 4)
**Goal**: Upload results and update Firestore

**Tasks**:
- [ ] Implement GCS uploader
- [ ] Generate clip metadata JSON
- [ ] Update Firestore with results
- [ ] Add thumbnail generation
- [ ] Implement task cleanup

**Deliverables**:
- Result upload system
- Firestore updates
- Metadata generation

---

### Phase 8: Testing & Optimization (Week 4-5)
**Goal**: Ensure reliability and performance

**Tasks**:
- [ ] Unit tests for all services
- [ ] Integration tests for full pipeline
- [ ] Load testing for concurrent tasks
- [ ] GPU memory optimization
- [ ] Error recovery testing
- [ ] Performance benchmarking

**Deliverables**:
- Comprehensive test suite
- Performance metrics
- Optimization report

---

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/test_transcription_processor.py
def test_segment_to_word_conversion():
    """Test conversion from segment to word-level timestamps"""
    segments = [
        {"start": 0.0, "end": 5.2, "text": "Hello world"}
    ]
    words = transcription_processor.segment_to_words(segments)
    assert len(words) == 2
    assert words[0]["word"] == "Hello"
    assert words[1]["word"] == "world"

# tests/test_caption_renderer.py
def test_caption_position_mapping():
    """Test ASS alignment value mapping"""
    assert caption_renderer.position_to_alignment("top-center") == 8
    assert caption_renderer.position_to_alignment("bottom-center") == 2
    assert caption_renderer.position_to_alignment("middle-center") == 5

# tests/test_video_processor.py
def test_ffmpeg_nvenc_available():
    """Ensure NVENC is available"""
    assert video_processor.check_nvenc_support() == True
```

### Integration Tests
```python
# tests/integration/test_full_pipeline.py
async def test_end_to_end_clip_generation():
    """Test complete clip generation pipeline"""
    task_id = "test-task-123"
    payload = {
        "task_id": task_id,
        "video_url": "gs://test-bucket/video.mp4",
        "audio_url": "gs://test-bucket/audio.mp3",
        "transcription": {"segments": [...]},
        "config": {"num_clips": 2}
    }
    
    # Process task
    result = await process_clip_task(payload)
    
    # Verify outputs
    assert result.status == "completed"
    assert len(result.clips) == 2
    assert all(clip.url.startswith("gs://") for clip in result.clips)
    
    # Verify Firestore update
    task_doc = firestore_client.collection("tasks").document(task_id).get()
    assert task_doc.exists
    assert task_doc.get("status") == "completed"
```

### Load Tests
```python
# tests/load/test_concurrent_tasks.py
async def test_concurrent_processing():
    """Test handling multiple tasks simultaneously"""
    tasks = [create_test_task() for _ in range(10)]
    results = await asyncio.gather(*[process_clip_task(t) for t in tasks])
    
    assert all(r.status == "completed" for r in results)
    # Verify no directory contamination
    assert len(set(r.task_id for r in results)) == 10
```

---

## ⚠️ Error Handling

### Error Categories

#### 1. Input Validation Errors
```python
class InvalidTaskPayloadError(Exception):
    """Raised when task payload is invalid"""
    pass

# Example
if not task_payload.task_id:
    raise InvalidTaskPayloadError("task_id is required")
```

#### 2. Asset Retrieval Errors
```python
class AssetDownloadError(Exception):
    """Raised when asset download fails"""
    pass

# Retry logic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def download_from_gcs(url: str, dest: str):
    try:
        # Download logic
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise AssetDownloadError(f"Download failed: {e}")
```

#### 3. Processing Errors
```python
class ClipGenerationError(Exception):
    """Raised when clip generation fails"""
    pass

# Graceful degradation
try:
    clips = clip_finder.find_clips(transcription)
    if not clips:
        # Fallback: divide video into equal segments
        clips = create_equal_segments(video_duration, num_clips)
except Exception as e:
    logger.error(f"ClipFinder failed: {e}")
    raise ClipGenerationError(f"Clip generation failed: {e}")
```

#### 4. FFmpeg Errors
```python
class VideoProcessingError(Exception):
    """Raised when FFmpeg processing fails"""
    pass

# Check exit codes
result = subprocess.run(ffmpeg_cmd, capture_output=True)
if result.returncode != 0:
    logger.error(f"FFmpeg error: {result.stderr.decode()}")
    raise VideoProcessingError(f"FFmpeg failed with code {result.returncode}")
```

### Error Recovery Strategy

```python
async def process_clip_task(task_payload: TaskPayload):
    try:
        # Update Firestore: status = "processing"
        await update_task_status(task_payload.task_id, "processing")
        
        # Phase 1: Asset Retrieval
        try:
            assets = await retrieve_assets(task_payload)
        except AssetDownloadError as e:
            await update_task_error(task_payload.task_id, f"Asset download failed: {e}")
            await cleanup_task(task_payload.task_id)
            return
        
        # Phase 2: Clip Generation
        try:
            clips = await generate_clips(assets, task_payload.config)
        except ClipGenerationError as e:
            # Fallback to equal segments
            clips = await create_fallback_clips(assets, task_payload.config)
        
        # Phase 3: Caption Rendering
        for clip in clips:
            try:
                await render_captions(clip, task_payload.config.subtitle_config)
            except VideoProcessingError as e:
                # Skip captions for this clip
                logger.warning(f"Skipping captions for clip {clip.id}: {e}")
                continue
        
        # Phase 4: Upload
        await upload_clips(clips, task_payload.task_id)
        
        # Success
        await update_task_status(task_payload.task_id, "completed")
        
    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Unexpected error processing task {task_payload.task_id}")
        await update_task_error(task_payload.task_id, str(e))
        
    finally:
        # Always cleanup
        await cleanup_task(task_payload.task_id)
```

---

## 🔒 Security Considerations

### 1. Input Validation
- Validate all URLs (must be gs:// from authorized bucket)
- Sanitize file paths to prevent directory traversal
- Validate task_id format (UUID only)
- Limit file sizes (max video size: 5GB)

### 2. Resource Limits
- Max concurrent tasks: 10
- Max processing time per task: 30 minutes
- Max GPU memory per task: 8GB
- Cleanup stale tasks after 1 hour

### 3. Authentication
- Require valid GCP service account
- Verify Pub/Sub message authenticity
- Use IAM for GCS/Firestore access

---

## 📊 Monitoring & Observability

### Metrics to Track
```python
from prometheus_client import Counter, Histogram, Gauge

# Task metrics
tasks_total = Counter('clips_tasks_total', 'Total tasks processed', ['status'])
task_duration = Histogram('clips_task_duration_seconds', 'Task processing duration')
tasks_active = Gauge('clips_tasks_active', 'Currently processing tasks')

# Clip metrics
clips_generated = Counter('clips_generated_total', 'Total clips generated')
clip_duration_avg = Gauge('clips_duration_avg_seconds', 'Average clip duration')

# Resource metrics
gpu_utilization = Gauge('clips_gpu_utilization', 'GPU utilization %')
ffmpeg_duration = Histogram('clips_ffmpeg_duration_seconds', 'FFmpeg processing time')

# Error metrics
errors_total = Counter('clips_errors_total', 'Total errors', ['error_type'])
```

### Logging Strategy
```python
import structlog

logger = structlog.get_logger()

# Structured logging
logger.info(
    "clip_generated",
    task_id=task_id,
    clip_id=clip_id,
    duration=clip.duration,
    engagement_score=clip.engagement_score,
    title=clip.title
)

# Error logging
logger.error(
    "ffmpeg_failed",
    task_id=task_id,
    command=" ".join(ffmpeg_cmd),
    exit_code=result.returncode,
    stderr=result.stderr.decode()
)
```

---

## 🚀 Deployment

### Cloud Run Configuration
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: clips-service
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"  # Disable CPU throttling
    spec:
      containerConcurrency: 1  # One request at a time
      timeoutSeconds: 1800     # 30 minutes
      containers:
      - image: gcr.io/{PROJECT_ID}/clips-service:latest
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: 16Gi
            cpu: "8"
        env:
        - name: GCS_BUCKET
          value: "your-clips-bucket"
        - name: FIRESTORE_PROJECT
          value: "your-project-id"
        - name: HUGGINGFACE_TOKEN
          valueFrom:
            secretKeyRef:
              name: clips-secrets
              key: huggingface-token
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: clips-secrets
              key: groq-api-key
```

### Build & Deploy Script
```bash
#!/bin/bash

# Build Docker image
gcloud builds submit --tag gcr.io/${PROJECT_ID}/clips-service:latest

# Deploy to Cloud Run
gcloud run deploy clips-service \
  --image gcr.io/${PROJECT_ID}/clips-service:latest \
  --platform managed \
  --region us-central1 \
  --gpu 1 \
  --gpu-type nvidia-tesla-t4 \
  --memory 16Gi \
  --cpu 8 \
  --timeout 1800 \
  --concurrency 1 \
  --min-instances 1 \
  --max-instances 10 \
  --no-allow-unauthenticated
```

---

## 📝 Sample Implementation Files

### `app/main.py`
```python
from fastapi import FastAPI, Request, HTTPException
from app.handlers.pubsub_handler import process_pubsub_message
from app.api.v1 import clips, health
import structlog

logger = structlog.get_logger()

app = FastAPI(
    title="Clips Service",
    description="AI-powered video clip generation with captions",
    version="1.0.0"
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(clips.router, prefix="/api/v1/clips", tags=["clips"])

@app.post("/")
async def pubsub_handler(request: Request):
    """Handle Pub/Sub push messages"""
    try:
        envelope = await request.json()
        await process_pubsub_message(envelope)
        return {"status": "accepted"}
    except Exception as e:
        logger.error("pubsub_handler_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Download models and initialize services"""
    logger.info("service_starting")
    # Download stable-ts model from GCS
    from app.services.transcription_processor import initialize_model
    await initialize_model()
    logger.info("service_ready")
```

### `app/services/caption_renderer.py`
```python
import os
from app.schemas import SubtitleConfig, CaptionPosition

POSITION_TO_ALIGNMENT = {
    CaptionPosition.TOP_LEFT: 7,
    CaptionPosition.TOP_CENTER: 8,
    CaptionPosition.TOP_RIGHT: 9,
    CaptionPosition.MIDDLE_LEFT: 4,
    CaptionPosition.MIDDLE_CENTER: 5,
    CaptionPosition.MIDDLE_RIGHT: 6,
    CaptionPosition.BOTTOM_LEFT: 1,
    CaptionPosition.BOTTOM_CENTER: 2,
    CaptionPosition.BOTTOM_RIGHT: 3,
}

def hex_to_ass_color(hex_color: str, alpha: int = 0) -> str:
    """Convert hex color to ASS format (BGR)"""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"

def generate_ass_file(
    word_info: list,
    clip_start: float,
    clip_end: float,
    config: SubtitleConfig,
    output_path: str
) -> str:
    """Generate ASS subtitle file with configuration"""
    
    # Build cues from word info
    cues = build_cues(word_info, clip_start, config.max_chars_per_line)
    
    # Get colors
    primary_color = hex_to_ass_color(config.font.color)
    accent_color = hex_to_ass_color(config.accent_color)
    outline_color = hex_to_ass_color(config.font.outline_color, alpha=0x40)
    
    # Get alignment
    alignment = POSITION_TO_ALIGNMENT[config.position]
    
    # Build ASS file
    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{config.font.family},{config.font.size},{primary_color},&H000000FF,{outline_color},&HFF000000,{-1 if config.font.bold else 0},{-1 if config.font.italic else 0},0,0,100,100,2,0,1,{config.font.outline_size},0,{alignment},30,30,{config.vertical_margin},1
Style: Yellow,{config.font.family},{config.font.size},{accent_color},&H000000FF,{outline_color},&HFF000000,{-1 if config.font.bold else 0},{-1 if config.font.italic else 0},0,0,100,100,2,0,1,{config.font.outline_size},0,{alignment},30,30,{config.vertical_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Add dialogue lines
    for cue in cues:
        start = format_ass_time(cue['start'])
        end = format_ass_time(cue['end'])
        text = format_text_with_colors(cue['text'])
        ass_content += f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
    
    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    return output_path

def build_cues(word_info: list, clip_start: float, max_chars: int) -> list:
    """Group words into subtitle cues"""
    # Implementation from main.py lines 124-165
    # ...
    pass

def format_text_with_colors(text: str) -> str:
    """Format text with accent colors for numbers"""
    words = text.split()
    formatted = []
    for word in words:
        if any(c.isdigit() for c in word) or '$' in word:
            formatted.append(f"{{\\rYellow}}{word}")
        else:
            formatted.append(word)
    return ' '.join(formatted)

def format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
```

---

## 🎯 Success Criteria

### Functional Requirements
- ✅ Process Pub/Sub messages successfully
- ✅ Download video/audio from GCS
- ✅ Generate word-level timestamps from segments
- ✅ Detect engaging clips using AI
- ✅ Render configurable captions
- ✅ Process videos with GPU acceleration
- ✅ Upload results to GCS
- ✅ Update Firestore with task status

### Performance Requirements
- ⚡ Process 5-minute video in < 3 minutes
- ⚡ Support 10 concurrent tasks
- ⚡ GPU utilization > 70%
- ⚡ Error rate < 1%
- ⚡ Task cleanup within 30 seconds

### Quality Requirements
- 🎥 Output videos: 1080x1920, H.264, 30fps
- 📝 Captions: Frame-accurate sync
- 🎨 Subtitle styling: User-configurable
- 🏆 Engagement scoring: > 0.6 threshold

---

## 📚 Additional Resources

### Documentation
- [ClipsAI GitHub](https://github.com/Zulko/clipsai)
- [stable-ts Documentation](https://github.com/jianfch/stable-ts)
- [FFmpeg NVENC Guide](https://developer.nvidia.com/ffmpeg)
- [ASS Subtitle Format](http://www.tcax.org/docs/ass-specs.htm)
- [FastAPI Cloud Run](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service)

### Example Commands

#### Test FFmpeg NVENC
```bash
ffmpeg -hwaccels  # Check for 'cuda'
ffmpeg -encoders | grep nvenc  # List NVENC encoders
```

#### Test stable-ts
```python
import stable_whisper

model = stable_whisper.load_model('medium')
result = model.transcribe(
    'audio.mp3',
    word_timestamps=True
)
print(result.to_dict())
```

#### Test GCS Download
```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('your-bucket')
blob = bucket.blob('input/video.mp4')
blob.download_to_filename('/tmp/video.mp4')
```

---

## 🏁 Next Steps

1. **Review this plan** with your team
2. **Set up development environment** with GPU access
3. **Create GCP resources** (bucket, Firestore, Pub/Sub topic)
4. **Start with Phase 1** (Core Infrastructure)
5. **Deploy to Cloud Run** for testing
6. **Iterate based on feedback**

---

## 📧 Support

For questions or issues during implementation:
- Check existing ClippedAI documentation
- Review FFmpeg/NVENC logs
- Monitor Cloud Run logs for errors
- Test each phase incrementally

**Good luck with the implementation! 🚀**

