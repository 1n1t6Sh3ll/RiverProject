<!DOCTYPE qgis>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
        <paletteEntry value="0" color="#000000" alpha="0" label="no water"/>
        <paletteEntry value="1" color="#2ca02c" alpha="170" label="1 usable"/>
        <paletteEntry value="2" color="#d62728" alpha="170" label="2 rejected: channel leaves cell"/>
        <paletteEntry value="3" color="#7f7f7f" alpha="170" label="3 rejected: other"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
