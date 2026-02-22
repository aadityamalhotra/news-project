import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import './NewsVisualization.css';

// Create convex hull-based bubble that encompasses all points in cluster
function createClusterHullGeometry(clusterArticles, centerX, centerY, centerZ) {
  const points = clusterArticles.map(a => ({
    x: a.x - centerX,
    y: a.y - centerY,
    z: a.z - centerZ
  }));

  const maxDistances = {
    x: Math.max(...points.map(p => Math.abs(p.x))),
    y: Math.max(...points.map(p => Math.abs(p.y))),
    z: Math.max(...points.map(p => Math.abs(p.z)))
  };

  const baseRadius = Math.max(maxDistances.x, maxDistances.y, maxDistances.z) * 1.5;
  const geometry = new THREE.SphereGeometry(baseRadius, 48, 48);
  const positions = geometry.attributes.position;

  const scaleX = maxDistances.x > 0 ? maxDistances.x / baseRadius : 0.5;
  const scaleY = maxDistances.y > 0 ? maxDistances.y / baseRadius : 0.5;
  const scaleZ = maxDistances.z > 0 ? maxDistances.z / baseRadius : 0.5;

  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i);
    const y = positions.getY(i);
    const z = positions.getZ(i);

    const length = Math.sqrt(x * x + y * y + z * z);
    const phi = Math.atan2(y, x);
    const theta = Math.acos(z / length);

    const deform = 1 +
      Math.sin(phi * 2) * Math.cos(theta * 2) * 0.15 +
      Math.sin(phi * 3 + theta * 2) * 0.10 +
      Math.cos(phi * 4) * Math.sin(theta * 3) * 0.08;

    const scale = deform * 1.6;
    positions.setXYZ(
      i,
      x * scaleX * scale,
      y * scaleY * scale,
      z * scaleZ * scale
    );
  }

  geometry.computeVertexNormals();
  return geometry;
}

// Coordinate axes component
function CoordinateAxes({ articles, articlesCenter }) {
  const axisLength = useMemo(() => {
    if (articles.length === 0) return 10;

    const maxDist = Math.max(...articles.map(a =>
      Math.sqrt(
        Math.pow(a.x - articlesCenter.x, 2) +
        Math.pow(a.y - articlesCenter.y, 2) +
        Math.pow(a.z - articlesCenter.z, 2)
      )
    ));

    return maxDist * 1.1;
  }, [articles, articlesCenter]);

  const axisColor = '#4A4947';

  return (
    <group position={[articlesCenter.x, articlesCenter.y, articlesCenter.z]}>
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([-axisLength, 0, 0, axisLength, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={axisColor} linewidth={1} />
      </line>

      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, -axisLength, 0, 0, axisLength, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={axisColor} linewidth={1} />
      </line>

      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, -axisLength, 0, 0, axisLength])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={axisColor} linewidth={1} />
      </line>
    </group>
  );
}

// Individual article point component
function ArticlePoint({ article, isSelected, isHighlighted, clusterColor, hoveredId, setHoveredId }) {
  const meshRef = useRef();
  const isThisHovered = hoveredId === article.article_id;

  useFrame(() => {
    if (meshRef.current) {
      if (isThisHovered) {
        meshRef.current.scale.setScalar(1.8 + Math.sin(Date.now() * 0.005) * 0.2);
      } else if (isHighlighted) {
        meshRef.current.scale.setScalar(1.2);
      } else {
        meshRef.current.scale.setScalar(1);
      }
    }
  });

  const handlePointerOver = (e) => {
    e.stopPropagation();
    setHoveredId(article.article_id);
  };

  const handlePointerOut = (e) => {
    e.stopPropagation();
    setHoveredId(null);
  };

  const color = useMemo(() => {
    if (!isHighlighted && isSelected) return '#808080';
    return clusterColor;
  }, [isHighlighted, isSelected, clusterColor]);

  const opacity = useMemo(() => {
    const isUnclustered = article.cluster_id === -1;
    if (isUnclustered) return isThisHovered ? 0.7 : 0.45;
    if (!isHighlighted && isSelected) return 0.2;
    return isThisHovered ? 1 : 0.85;
  }, [isHighlighted, isSelected, isThisHovered, article.cluster_id]);

  return (
    <group position={[article.x, article.y, article.z]}>
      <mesh
        ref={meshRef}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
      >
        <sphereGeometry args={[0.18, 16, 16]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={opacity}
          emissive={color}
          emissiveIntensity={isThisHovered ? 0.7 : 0.25}
          metalness={0.4}
          roughness={0.6}
        />
      </mesh>

      {isThisHovered && (
        <Html
          position={[0, 0.5, 0]}
          style={{
            pointerEvents: 'none',
            width: '320px',
            transform: 'translate(-50%, -100%)'
          }}
        >
          <div className="article-tooltip">
            <div className="tooltip-title">
              {article.title.split(' ').slice(0, 10).join(' ')}
              {article.title.split(' ').length > 10 ? '...' : ''}
            </div>
            <div className="tooltip-source">{article.source_name}</div>
          </div>
        </Html>
      )}
    </group>
  );
}

// Cluster bubble component
function ClusterBubble({ cluster, articles, isSelected, isHighlighted }) {
  const meshRef = useRef();
  const [geometry, setGeometry] = useState(null);

  useEffect(() => {
    const clusterArticles = articles.filter(a => a.cluster_id === cluster.cluster_id);
    if (clusterArticles.length === 0) return;

    const hullGeometry = createClusterHullGeometry(
      clusterArticles,
      cluster.center_x,
      cluster.center_y,
      cluster.center_z
    );
    setGeometry(hullGeometry);

    return () => {
      if (hullGeometry) hullGeometry.dispose();
    };
  }, [cluster, articles]);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.0008;
      meshRef.current.rotation.x += 0.0003;

      const breathe = 1 + Math.sin(state.clock.elapsedTime * 0.4) * 0.03;
      const targetScale = isHighlighted ? 1.08 * breathe : 1.0 * breathe;
      meshRef.current.scale.lerp(
        new THREE.Vector3(targetScale, targetScale, targetScale),
        0.05
      );
    }
  });

  const opacity = useMemo(() => {
    if (!isHighlighted && isSelected) return 0.03;
    return isHighlighted ? 0.28 : 0.16;
  }, [isHighlighted, isSelected]);

  const color = useMemo(() => {
    if (!isHighlighted && isSelected) return '#808080';
    return cluster.color;
  }, [isHighlighted, isSelected, cluster.color]);

  if (!geometry) return null;

  return (
    <mesh
      ref={meshRef}
      position={[cluster.center_x, cluster.center_y, cluster.center_z]}
      geometry={geometry}
      renderOrder={-1}
    >
      <meshStandardMaterial
        color={color}
        transparent
        opacity={opacity}
        emissive={color}
        emissiveIntensity={0.5}
        wireframe={false}
        side={THREE.DoubleSide}
        depthWrite={false}
        metalness={0.2}
        roughness={0.8}
      />
    </mesh>
  );
}

// Main scene component
function Scene({ articles, clusters, selectedCluster }) {
  const controlsRef = useRef();
  const [hoveredId, setHoveredId] = useState(null);

  const articlesCenter = useMemo(() => {
    if (articles.length === 0) return new THREE.Vector3(0, 0, 0);
    return new THREE.Vector3(
      articles.reduce((sum, a) => sum + a.x, 0) / articles.length,
      articles.reduce((sum, a) => sum + a.y, 0) / articles.length,
      articles.reduce((sum, a) => sum + a.z, 0) / articles.length
    );
  }, [articles]);

  useEffect(() => {
    if (controlsRef.current && articles.length > 0) {
      controlsRef.current.target.copy(articlesCenter);
      controlsRef.current.update();
    }
  }, [articlesCenter, articles.length]);

  useEffect(() => {
    if (selectedCluster !== null && controlsRef.current) {
      const cluster = clusters.find(c => c.cluster_id === selectedCluster);
      if (cluster) {
        controlsRef.current.target.copy(
          new THREE.Vector3(cluster.center_x, cluster.center_y, cluster.center_z)
        );
        controlsRef.current.update();
      }
    } else if (selectedCluster === null && controlsRef.current) {
      controlsRef.current.target.copy(articlesCenter);
      controlsRef.current.update();
    }
  }, [selectedCluster, clusters, articlesCenter]);

  return (
    <>
      <ambientLight intensity={0.7} />
      <pointLight position={[15, 15, 15]} intensity={1.0} color="#FAF7F0" />
      <pointLight position={[-15, -15, -15]} intensity={0.5} color="#D8D2C2" />
      <directionalLight position={[5, 5, 10]} intensity={0.4} color="#FAF7F0" />

      <CoordinateAxes articles={articles} articlesCenter={articlesCenter} />

      {clusters.filter(c => c.cluster_id !== -1).map(cluster => (
        <ClusterBubble
          key={cluster.cluster_id}
          cluster={cluster}
          articles={articles}
          isSelected={selectedCluster !== null}
          isHighlighted={selectedCluster === null || selectedCluster === cluster.cluster_id}
        />
      ))}

      {articles.map(article => {
        const cluster = clusters.find(c => c.cluster_id === article.cluster_id);
        const isUncategorized = article.cluster_id === -1;
        return (
          <ArticlePoint
            key={article.article_id}
            article={article}
            isSelected={selectedCluster !== null}
            isHighlighted={selectedCluster === null || selectedCluster === article.cluster_id}
            clusterColor={isUncategorized ? '#808080' : (cluster?.color || '#667eea')}
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
          />
        );
      })}

      <OrbitControls
        ref={controlsRef}
        target={articlesCenter}
        enableDamping={true}
        dampingFactor={0.08}
        rotateSpeed={0.5}
        zoomSpeed={0.7}
        panSpeed={0.5}
        minDistance={8}
        maxDistance={45}
        enablePan={true}
        mouseButtons={{
          LEFT: THREE.MOUSE.ROTATE,
          MIDDLE: THREE.MOUSE.DOLLY,
          RIGHT: THREE.MOUSE.PAN
        }}
      />
    </>
  );
}

export default function NewsVisualization({ articles, clusters, selectedCluster }) {
  return (
    <div className="news-visualization">
      <Canvas
        camera={{ position: [0, 0, 22], fov: 55 }}
        gl={{ antialias: true, alpha: true }}
      >
        <Scene
          articles={articles}
          clusters={clusters}
          selectedCluster={selectedCluster}
        />
      </Canvas>

      <div className="visualization-info">
        <div className="info-item">
          <span className="info-label">Articles</span>
          <span className="info-value">{articles.length}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Clusters</span>
          <span className="info-value">{clusters.filter(c => c.cluster_id !== -1).length}</span>
        </div>
      </div>

      <div className="visualization-controls">
        {/* Fixed: was garbled mojibake from Latin-1/UTF-8 mismatch */}
        <p>Drag to rotate &bull; Scroll to zoom &bull; Hover for details</p>
      </div>
    </div>
  );
}
