xof 0303txt 0032

Frame Root {
  FrameTransformMatrix {
     1.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 1.000000, 0.000000, 0.000000,
     0.000000, 0.000000, 1.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 1.000000;;
  }
  Frame Cube {
    FrameTransformMatrix {
       1.000000, 0.000000, 0.000000, 0.000000,
       0.000000, 1.000000, 0.000000, 0.000000,
       0.000000, 0.000000, 1.000000, 0.000000,
       0.000000, 0.000000, 0.000000, 1.000000;;
    }
    Mesh { // Cube mesh
      6;
       0.100000; 0.100000;-0.100000;,
       0.100000; 0.099999; 0.100000;,
       0.100000;-0.100000;-0.100000;,
       0.100000; 0.099999; 0.100000;,
       0.099999;-0.100001; 0.100000;,
       0.100000;-0.100000;-0.100000;;
      2;
      3;2,1,0;,
      3;5,4,3;;
      MeshNormals { // Cube normals
        6;
        -1.000000; 0.000000; 0.000000;,
        -1.000000; 0.000000; 0.000000;,
        -1.000000; 0.000000; 0.000000;,
        -1.000000; 0.000000; 0.000000;,
        -1.000000; 0.000000; 0.000000;,
        -1.000000; 0.000000; 0.000000;;
      } // End of Cube normals
      MeshTextureCoords { // Cube UV coordinates
        6;
         0.999900; 0.999900;,
         0.999900; 0.000100;,
         0.000100; 0.999900;,
         0.999900; 0.000100;,
         0.000100; 0.000100;,
         0.000100; 0.999900;;
      } // End of Cube UV coordinates
      MeshMaterialList { // Cube material list
        1;
        2;
        0,
        0;;
        Material Material {
           0.640000; 0.640000; 0.640000; 1.000000;;
           96.078431;
           0.500000; 0.500000; 0.500000;;
           0.000000; 0.000000; 0.000000;;
          TextureFilename {"marker_7.bmp";}
        }
      } // End of Cube material list
    } // End of Cube mesh
  } // End of Cube
} // End of Root
