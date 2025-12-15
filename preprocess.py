"""
TSP 데이터 전처리 통합 파이프라인
=================================
원본 배송 데이터를 TSP 모델 학습용 데이터셋으로 변환

주요 변경사항:
- 노드 순서를 랜덤하게 섞어서 실제 TSP 문제로 만듦
- 원래 시간순 배송 순서가 정답 경로(actual_route)가 됨
- 모든 전처리 단계를 하나의 파일로 통합

사용법:
    python preprocess.py --input delivery_sh-*.parquet --output ./model_data/
"""

import pandas as pd
import numpy as np
import pickle
import re
import random
import os
import argparse
from math import radians, cos, sin, asin, sqrt
from datetime import timedelta
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ==========================================
# 설정 (Configuration)
# ==========================================
class Config:
    # 데이터 필터링
    TIME_GAP_MINUTES = 30      # Trip 분할 기준 (분)
    MIN_NODES = 5              # 최소 노드 수
    MAX_NODES = 100            # 최대 노드 수
    
    # 좌표 범위 (상하이)
    LAT_MIN, LAT_MAX = 30.5, 32.0
    LNG_MIN, LNG_MAX = 120.8, 122.0
    
    # 교통 속도 추출
    TORTUOSITY_FACTOR = 1.4    # 도로 우회 계수
    SERVICE_TIME_SEC = 180     # 배송 서비스 시간 (초)
    MIN_SPEED_KMH = 5.0        # 최소 유효 속도
    MAX_SPEED_KMH = 60.0       # 최대 유효 속도
    SPEED_SAMPLE_SIZE = 20000  # 속도 추출용 샘플 크기
    
    # 데이터 분할
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    RANDOM_SEED = 42
    
    # 연도 (데이터셋에 연도 정보가 없을 때)
    DEFAULT_YEAR = 2023


# ==========================================
# 유틸리티 함수
# ==========================================
def haversine(lon1, lat1, lon2, lat2):
    """두 GPS 좌표 간 직선 거리 (km)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    return c * 6371


def parse_time_with_mmdd_format(delivery_time, ds, default_year=2023):
    """
    ds가 'MMdd' 형식 (예: '604' = 6월 4일)일 때 처리
    """
    try:
        if pd.isna(delivery_time) or pd.isna(ds):
            return pd.NaT
        
        ds_str = str(ds).zfill(4)  # '604' → '0604'
        month = ds_str[:2]
        day = ds_str[2:]
        
        time_str = str(delivery_time).strip()
        match = re.search(r'(\d{2}):(\d{2}):(\d{2})', time_str)
        
        if match:
            time_part = match.group(0)
            full_datetime = f"{default_year}-{month}-{day} {time_part}"
            return pd.to_datetime(full_datetime, format='%Y-%m-%d %H:%M:%S')
        
        return pd.NaT
        
    except Exception:
        return pd.NaT


# ==========================================
# Step 1: 원시 데이터 → Trip 생성
# ==========================================
def step1_create_trips(input_file, config=Config):
    """
    원시 배송 데이터에서 Trip 단위로 묶기
    """
    print("\n" + "="*60)
    print("📦 Step 1: Trip 생성")
    print("="*60)
    
    # 데이터 로드
    print(f"\n📂 데이터 로드 중: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"   - 총 배송 건수: {len(df):,}")
    
    # 시간 변환
    print("\n🔄 시간 변환 중...")
    df['delivery_dt'] = df.apply(
        lambda row: parse_time_with_mmdd_format(
            row['delivery_time'], row['ds'], config.DEFAULT_YEAR
        ),
        axis=1
    )
    
    # 변환 실패 제거
    null_count = df['delivery_dt'].isna().sum()
    if null_count > 0:
        print(f"   ⚠️ 시간 변환 실패: {null_count:,}개 제거")
        df = df.dropna(subset=['delivery_dt']).reset_index(drop=True)
    
    print(f"   ✅ 변환 완료: {len(df):,}개")
    
    # 좌표 검증
    print("\n🌍 좌표 검증 중...")
    initial_rows = len(df)
    df = df[
        (df['delivery_gps_lat'].between(config.LAT_MIN, config.LAT_MAX)) &
        (df['delivery_gps_lng'].between(config.LNG_MIN, config.LNG_MAX))
    ].reset_index(drop=True)
    removed = initial_rows - len(df)
    print(f"   - 이상 좌표 제거: {removed:,}개")
    
    # 정렬 (기사 → 시간순)
    df = df.sort_values(['courier_id', 'delivery_dt']).reset_index(drop=True)
    
    # Trip 분할
    print("\n✂️ Trip 분할 중...")
    df['prev_courier'] = df['courier_id'].shift(1)
    df['prev_delivery_dt'] = df['delivery_dt'].shift(1)
    df['time_gap_min'] = (
        (df['delivery_dt'] - df['prev_delivery_dt']).dt.total_seconds() / 60.0
    )
    df['is_new_trip'] = (
        (df['courier_id'] != df['prev_courier']) | 
        (df['time_gap_min'] > config.TIME_GAP_MINUTES) | 
        (df['prev_courier'].isna())
    )
    df['trip_id'] = df['is_new_trip'].cumsum()
    
    # Trip별 집계
    print("📦 Trip 집계 중...")
    trip_df = df.groupby('trip_id').agg({
        'courier_id': 'first',
        'delivery_gps_lat': list,
        'delivery_gps_lng': list,
        'delivery_dt': list,
        'region_id': list,
        'order_id': lambda x: list(x),
        'ds': 'first',
        'city': 'first'
    }).rename(columns={'order_id': 'order_ids'})
    
    trip_df['num_nodes'] = trip_df['delivery_gps_lat'].apply(len)
    
    # 필터링
    initial_count = len(trip_df)
    trip_df = trip_df[
        (trip_df['num_nodes'] >= config.MIN_NODES) & 
        (trip_df['num_nodes'] <= config.MAX_NODES)
    ].reset_index(drop=True)
    
    print(f"   - 필터링 전: {initial_count:,}개")
    print(f"   - 필터링 후: {len(trip_df):,}개")
    
    # 구간별 시간 계산
    def calculate_segment_times(delivery_times):
        times = []
        for i in range(len(delivery_times) - 1):
            duration = (delivery_times[i+1] - delivery_times[i]).total_seconds()
            times.append(max(duration, 1.0))
        return times
    
    trip_df['actual_segment_times'] = trip_df['delivery_dt'].apply(calculate_segment_times)
    trip_df['actual_total_time'] = trip_df['actual_segment_times'].apply(sum)
    
    print(f"\n✅ Step 1 완료: {len(trip_df):,}개 Trip")
    
    return trip_df


# ==========================================
# Step 2: 노드 순서 셔플 (핵심 수정!)
# ==========================================
def step2_shuffle_nodes(trip_df, random_seed=42):
    """
    🔥 핵심 변경: 노드 순서를 랜덤하게 섞어서 실제 TSP 문제로 만듦
    
    - 입력 좌표: 랜덤 순서로 섞인 노드들
    - 정답 경로: 섞인 좌표에서 원래 시간순 배송 순서를 복원하는 인덱스
    
    예시:
        원래 시간순: [A, B, C, D] (인덱스 0, 1, 2, 3)
        셔플 후:     [C, A, D, B] (인덱스 0, 1, 2, 3 = 원래 2, 0, 3, 1)
        정답 경로:   [1, 3, 0, 2] (셔플된 배열에서 A→B→C→D 순서)
    """
    print("\n" + "="*60)
    print("🔀 Step 2: 노드 순서 셔플 (TSP 문제 생성)")
    print("="*60)
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    shuffled_data = []
    
    for idx, row in tqdm(trip_df.iterrows(), total=len(trip_df), desc="노드 셔플"):
        n = row['num_nodes']
        
        # 원래 데이터 (시간순 정렬)
        orig_lats = list(row['delivery_gps_lat'])
        orig_lngs = list(row['delivery_gps_lng'])
        orig_times = list(row['delivery_dt'])
        orig_segment_times = list(row['actual_segment_times'])
        
        # 셔플 인덱스 생성 (depot인 0번은 고정, 나머지만 셔플)
        shuffled_indices = [0] + random.sample(range(1, n), n - 1)
        
        # 좌표와 시간을 셔플된 순서로 재배치
        shuffled_lats = [orig_lats[i] for i in shuffled_indices]
        shuffled_lngs = [orig_lngs[i] for i in shuffled_indices]
        shuffled_times = [orig_times[i] for i in shuffled_indices]
        
        # 🔥 정답 경로 계산: 셔플된 배열에서 원래 시간순을 복원하는 인덱스
        # inverse_map: 원래 인덱스 → 셔플된 배열에서의 위치
        inverse_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(shuffled_indices)}
        actual_route = [inverse_map[orig_idx] for orig_idx in range(n)]
        
        # 구간 시간도 정답 경로 순서에 맞게 재배치
        # actual_route[i] → actual_route[i+1] 이동 시간 = 원래 구간 시간
        # (실제로는 원래 시간순 구간 시간을 그대로 사용)
        
        shuffled_data.append({
            'courier_id': row['courier_id'],
            'delivery_gps_lat': shuffled_lats,
            'delivery_gps_lng': shuffled_lngs,
            'delivery_dt': shuffled_times,
            'region_id': list(row['region_id']) if isinstance(row['region_id'], (list, np.ndarray)) else row['region_id'],
            'order_ids': row['order_ids'],
            'ds': row['ds'],
            'city': row['city'],
            'num_nodes': n,
            'actual_route': actual_route,
            'actual_segment_times': orig_segment_times,  # 원래 시간순 구간 시간
            'actual_total_time': row['actual_total_time'],
            'shuffle_indices': shuffled_indices  # 디버깅용
        })
    
    result_df = pd.DataFrame(shuffled_data)
    
    # 검증: 정답 경로가 더 이상 순차적이지 않은지 확인
    sequential_count = 0
    for _, row in result_df.iterrows():
        if row['actual_route'] == list(range(row['num_nodes'])):
            sequential_count += 1
    
    print(f"\n📊 셔플 검증:")
    print(f"   - 전체 Trip: {len(result_df):,}개")
    print(f"   - 순차적 경로 (셔플 안됨): {sequential_count}개")
    print(f"   - 정상 셔플: {len(result_df) - sequential_count:,}개")
    
    # 샘플 확인
    sample = result_df.iloc[0]
    print(f"\n🔎 샘플 확인:")
    print(f"   - 노드 수: {sample['num_nodes']}")
    print(f"   - 셔플 인덱스: {sample['shuffle_indices'][:10]}...")
    print(f"   - 정답 경로: {sample['actual_route'][:10]}...")
    
    print(f"\n✅ Step 2 완료: 노드 순서 셔플 적용됨")
    
    return result_df


# ==========================================
# Step 3: 교통 속도 프로필 추출 & Baseline
# ==========================================
def step3_traffic_and_baseline(trip_df, config=Config):
    """
    시간대별 교통 속도 프로필 추출 및 Greedy Baseline 생성
    """
    print("\n" + "="*60)
    print("🚦 Step 3: 교통 패턴 추출 & Baseline 생성")
    print("="*60)
    
    # 속도 데이터 추출
    print("\n📊 시간대별 속도 추출 중...")
    
    speed_data = []
    sample_size = min(len(trip_df), config.SPEED_SAMPLE_SIZE)
    sample_df = trip_df.sample(sample_size, random_state=config.RANDOM_SEED)
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="속도 추출"):
        lats = row['delivery_gps_lat']
        lngs = row['delivery_gps_lng']
        times = row['delivery_dt']
        actual_route = row['actual_route']
        
        # 정답 경로 순서대로 속도 계산
        for step in range(len(actual_route) - 1):
            try:
                curr_idx = actual_route[step]
                next_idx = actual_route[step + 1]
                
                # 시간 차이
                t1 = pd.Timestamp(times[curr_idx])
                t2 = pd.Timestamp(times[next_idx])
                total_duration_sec = (t2 - t1).total_seconds()
                
                if total_duration_sec <= 0:
                    continue
                
                drive_duration_sec = max(total_duration_sec - config.SERVICE_TIME_SEC, 10)
                
                # 거리
                dist_km = haversine(
                    lngs[curr_idx], lats[curr_idx],
                    lngs[next_idx], lats[next_idx]
                ) * config.TORTUOSITY_FACTOR
                
                if dist_km < 0.1:
                    continue
                
                # 속도
                speed_kmh = (dist_km / drive_duration_sec) * 3600
                
                if config.MIN_SPEED_KMH < speed_kmh < config.MAX_SPEED_KMH:
                    speed_data.append({
                        'hour': t1.hour,
                        'speed': speed_kmh,
                        'distance': dist_km
                    })
                    
            except Exception:
                continue
    
    print(f"   - 추출된 속도 샘플: {len(speed_data):,}개")
    
    # 속도 프로필 생성
    if len(speed_data) > 0:
        speed_df = pd.DataFrame(speed_data)
        speed_profile = (
            speed_df.groupby('hour')
            .apply(lambda x: np.average(x['speed'], weights=x['distance']))
            .reindex(range(24))
            .interpolate(method='linear')
            .fillna(method='bfill')
            .fillna(method='ffill')
            .to_dict()
        )
    else:
        # 기본 속도 프로필 (추출 실패 시)
        print("   ⚠️ 속도 추출 실패, 기본값 사용")
        speed_profile = {h: 20.0 for h in range(24)}
    
    print("\n📊 시간대별 평균 속도 (km/h):")
    for h in [6, 9, 12, 15, 18, 21]:
        bar = '█' * int(speed_profile[h] / 2)
        print(f"   {h:02d}시: {speed_profile[h]:5.1f} {bar}")
    
    # Greedy Baseline 생성
    print("\n🏃 Greedy Baseline 생성 중...")
    
    def greedy_baseline(lats, lngs, start_time, speed_profile):
        """셔플된 좌표에서 Greedy 경로 생성"""
        n = len(lats)
        unvisited = set(range(1, n))
        route = [0]
        current = 0
        current_time = pd.Timestamp(start_time)
        segment_times = []
        
        while unvisited:
            hour = current_time.hour
            speed_kmh = speed_profile[hour]
            speed_mps = (speed_kmh * 1000) / 3600
            
            best_node = None
            min_travel_time = float('inf')
            
            for node in unvisited:
                dist_km = haversine(lngs[current], lats[current], lngs[node], lats[node])
                dist_m = dist_km * config.TORTUOSITY_FACTOR * 1000
                travel_sec = dist_m / speed_mps
                
                if travel_sec < min_travel_time:
                    min_travel_time = travel_sec
                    best_node = node
            
            step_duration = min_travel_time + config.SERVICE_TIME_SEC
            segment_times.append(step_duration)
            current_time += timedelta(seconds=step_duration)
            
            route.append(best_node)
            unvisited.remove(best_node)
            current = best_node
        
        return route, segment_times
    
    baseline_routes = []
    baseline_segment_times = []
    
    for _, row in tqdm(trip_df.iterrows(), total=len(trip_df), desc="Baseline 생성"):
        route, seg_times = greedy_baseline(
            row['delivery_gps_lat'],
            row['delivery_gps_lng'],
            row['delivery_dt'][0],
            speed_profile
        )
        baseline_routes.append(route)
        baseline_segment_times.append(seg_times)
    
    trip_df['baseline_route'] = baseline_routes
    trip_df['baseline_segment_times'] = baseline_segment_times
    trip_df['baseline_total_time'] = [sum(times) for times in baseline_segment_times]
    
    # 속도 프로필 저장
    speed_profile_24h = [speed_profile[h] for h in range(24)]
    trip_df['speed_profile_24h'] = [speed_profile_24h] * len(trip_df)
    
    # 성능 비교
    print("\n📊 Baseline vs 실제 경로:")
    actual_avg = trip_df['actual_total_time'].mean()
    baseline_avg = trip_df['baseline_total_time'].mean()
    print(f"   - 실제 기사: {actual_avg/60:.1f}분")
    print(f"   - Baseline:  {baseline_avg/60:.1f}분")
    
    print(f"\n✅ Step 3 완료")
    
    return trip_df, speed_profile


# ==========================================
# Step 4: 모델 입력 형태로 변환 & 저장
# ==========================================
def step4_prepare_and_save(trip_df, output_dir, config=Config):
    """
    데이터 분할 및 모델 입력 형태로 변환
    """
    print("\n" + "="*60)
    print("💾 Step 4: 데이터 분할 & 저장")
    print("="*60)
    
    # 좌표 정규화 범위 계산
    print("\n🌍 좌표 정규화 범위 계산 중...")
    all_lats = []
    all_lngs = []
    for _, row in trip_df.iterrows():
        all_lats.extend(row['delivery_gps_lat'])
        all_lngs.extend(row['delivery_gps_lng'])
    
    lat_min, lat_max = min(all_lats), max(all_lats)
    lng_min, lng_max = min(all_lngs), max(all_lngs)
    
    print(f"   - 위도: {lat_min:.4f} ~ {lat_max:.4f}")
    print(f"   - 경도: {lng_min:.4f} ~ {lng_max:.4f}")
    
    def normalize_coords(lats, lngs):
        norm_lats = [(lat - lat_min) / (lat_max - lat_min + 1e-6) for lat in lats]
        norm_lngs = [(lng - lng_min) / (lng_max - lng_min + 1e-6) for lng in lngs]
        return norm_lats, norm_lngs
    
    # 데이터 분할
    print("\n✂️ 데이터 분할 중...")
    train_val_df, test_df = train_test_split(
        trip_df, test_size=config.TEST_RATIO, random_state=config.RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO),
        random_state=config.RANDOM_SEED
    )
    
    print(f"   - Train: {len(train_df):,}개")
    print(f"   - Val:   {len(val_df):,}개")
    print(f"   - Test:  {len(test_df):,}개")
    
    # 모델 입력 형태로 변환
    def prepare_model_input(row):
        lats = row['delivery_gps_lat']
        lngs = row['delivery_gps_lng']
        
        norm_lats, norm_lngs = normalize_coords(lats, lngs)
        node_features = np.column_stack([norm_lats, norm_lngs]).astype(np.float32)
        
        # 시간 정보
        start_time = pd.Timestamp(row['delivery_dt'][0])
        
        return {
            # 입력
            'node_features': node_features,
            'start_hour': start_time.hour,
            'start_minute': start_time.minute,
            'day_of_week': start_time.weekday(),
            'traffic_profile': np.array(row['speed_profile_24h'], dtype=np.float32),
            'num_nodes': len(lats),
            
            # 정답 라벨 (셔플된 좌표 기준)
            'actual_route': np.array(row['actual_route'], dtype=np.int32),
            'actual_times': np.array(row['actual_segment_times'], dtype=np.float32),
            'actual_total_time': row['actual_total_time'],
            
            # Baseline
            'baseline_route': np.array(row['baseline_route'], dtype=np.int32),
            'baseline_times': np.array(row['baseline_segment_times'], dtype=np.float32),
            'baseline_total_time': row['baseline_total_time'],
            
            # 메타
            'courier_id': row['courier_id'],
            'trip_id': row.name
        }
    
    print("\n🔄 모델 입력 형태로 변환 중...")
    train_data = [prepare_model_input(row) for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train")]
    val_data = [prepare_model_input(row) for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val")]
    test_data = [prepare_model_input(row) for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test")]
    
    # 저장
    print("\n💾 저장 중...")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(f'{output_dir}/val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(f'{output_dir}/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    # 메타데이터
    metadata = {
        'num_train': len(train_data),
        'num_val': len(val_data),
        'num_test': len(test_data),
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lng_min': lng_min,
        'lng_max': lng_max,
        'avg_nodes': trip_df['num_nodes'].mean(),
        'max_nodes': trip_df['num_nodes'].max(),
        'min_nodes': trip_df['num_nodes'].min(),
    }
    
    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n   ✅ train_data.pkl ({len(train_data):,}개)")
    print(f"   ✅ val_data.pkl ({len(val_data):,}개)")
    print(f"   ✅ test_data.pkl ({len(test_data):,}개)")
    print(f"   ✅ metadata.pkl")
    
    # 샘플 확인
    print("\n🔎 샘플 데이터 구조:")
    sample = train_data[0]
    print(f"   - node_features: shape {sample['node_features'].shape}")
    print(f"   - actual_route: {sample['actual_route'][:10]}... (비순차적!)")
    print(f"   - baseline_route: {sample['baseline_route'][:10]}...")
    print(f"   - num_nodes: {sample['num_nodes']}")
    
    print(f"\n✅ Step 4 완료")
    
    return train_data, val_data, test_data, metadata


# ==========================================
# 메인 파이프라인
# ==========================================
def run_pipeline(input_file, output_dir='./model_data/'):
    """
    전체 전처리 파이프라인 실행
    """
    print("\n" + "="*70)
    print("🚀 TSP 데이터 전처리 파이프라인 시작")
    print("="*70)
    print(f"   입력: {input_file}")
    print(f"   출력: {output_dir}")
    print("="*70)
    
    # Step 1: Trip 생성
    trip_df = step1_create_trips(input_file)
    
    # Step 2: 노드 순서 셔플 (핵심!)
    trip_df = step2_shuffle_nodes(trip_df)
    
    # Step 3: 교통 패턴 & Baseline
    trip_df, speed_profile = step3_traffic_and_baseline(trip_df)
    
    # Step 4: 저장
    train_data, val_data, test_data, metadata = step4_prepare_and_save(trip_df, output_dir)
    
    # 완료
    print("\n" + "="*70)
    print("🎉 전처리 완료!")
    print("="*70)
    print(f"\n📦 생성된 파일:")
    print(f"   - {output_dir}/train_data.pkl ({len(train_data):,}개)")
    print(f"   - {output_dir}/val_data.pkl ({len(val_data):,}개)")
    print(f"   - {output_dir}/test_data.pkl ({len(test_data):,}개)")
    print(f"   - {output_dir}/metadata.pkl")
    print("\n✅ 다음 단계: python train.py")
    print("="*70)
    
    return trip_df


# ==========================================
# CLI 실행
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TSP 데이터 전처리')
    parser.add_argument('--input', '-i', type=str, 
                        default='delivery_sh-00000-of-00001-ad9a4b1d79823540.parquet',
                        help='입력 parquet 파일')
    parser.add_argument('--output', '-o', type=str,
                        default='./model_data/',
                        help='출력 디렉토리')
    
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output)

