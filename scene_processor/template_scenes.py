from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class TemplateObject:
    """템플릿 씬의 오브젝트 정의."""
    mesh_path: str
    """템플릿 메시 파일 경로 (examples/templates 기준 상대 경로)"""
    translation: List[float]
    """위치 [x, y, z]"""
    rotation: List[float]
    """회전 [x, y, z] (도)"""
    scale: List[float]
    """스케일 [x, y, z]"""
    normalize: bool = False
    """단위 구 정규화 여부"""

@dataclass
class TemplateScene:
    """템플릿 씬 정의."""
    template_id: int
    """템플릿 ID (0-3)"""
    name: str
    """템플릿 이름"""
    description: str
    """템플릿 설명"""
    background_objects: List[TemplateObject]
    """배경 오브젝트들 (바닥, 벽 등)"""
    template_dir: str = "templates"
    """템플릿 디렉토리 경로 (examples 기준)"""

TEMPLATE_SCENES: Dict[int, TemplateScene] = {
    0: TemplateScene(
        template_id=0,
        name="Open Space",
        description="No walls",
        background_objects=[
            TemplateObject(
                mesh_path="templates/backgrounds/plane.obj",
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.5, 0.5, 0.5],
                normalize=False
            ),
        ]
    ),
    1: TemplateScene(
        template_id=1,
        name="Back Wall",
        description="Back Wall",
        background_objects=[
            TemplateObject(
                mesh_path="templates/backgrounds/plane.obj",
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.5, 0.5, 0.5],
                normalize=False
            ),
            TemplateObject(
                mesh_path="templates/backgrounds/wall0.obj",
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.5, 0.5, 0.5],
                normalize=False
            ),
        ]
    ),
    2: TemplateScene(
        template_id=2,
        name="Two Walls",
        description="Two Walls",
        background_objects=[
            TemplateObject(
                mesh_path="templates/backgrounds/plane.obj",
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.5, 0.5, 0.5],
                normalize=False
            ),
            TemplateObject(
                mesh_path="templates/backgrounds/wall0.obj",
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.5, 0.5, 0.5],
                normalize=False
            ),
            TemplateObject(
                mesh_path="templates/backgrounds/wall1.obj",
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.5, 0.5, 0.5],
                normalize=False
            ),
        ]
    ),
    3: TemplateScene(
        template_id=3,
        name="Three Walls",
        description="Three Walls",
        background_objects=[
            TemplateObject(
                mesh_path="templates/backgrounds/plane.obj",
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.5, 0.5, 0.5],
                normalize=False
            ),
            TemplateObject(
                mesh_path="templates/backgrounds/wall0.obj",
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.5, 0.5, 0.5],
                normalize=False
            ),
            TemplateObject(
                mesh_path="templates/backgrounds/wall1.obj",
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.5, 0.5, 0.5],
                normalize=False
            ),
            TemplateObject(
                mesh_path="templates/backgrounds/wall2.obj",
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[0.5, 0.5, 0.5],
                normalize=False
            ),
        ]
    ),
}

def get_template_scene(template_id: int) -> TemplateScene:
    """템플릿 씬을 가져옵니다.
    
    Args:
        template_id:
        
    Returns:
        
    Raises:
        ValueError:
    """
    if template_id not in TEMPLATE_SCENES:
        raise ValueError(f"Invalid template_id: {template_id}. Must be 0-3.")
    return TEMPLATE_SCENES[template_id]

def get_all_template_ids() -> List[int]:
    """모든 템플릿 ID 리스트를 반환합니다."""
    return list(TEMPLATE_SCENES.keys())
