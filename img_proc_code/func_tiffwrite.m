% Save single data into tif image
function func_tiffwrite(saveimg, savepath)
    t = Tiff(savepath, 'w');
    tagstruct.ImageLength = size(saveimg, 1);
    tagstruct.ImageWidth = size(saveimg, 2);
    tagstruct.Compression = Tiff.Compression.None;
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tagstruct.Photometric = Tiff.Photometric.LinearRaw;
    tagstruct.BitsPerSample = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    t.setTag(tagstruct);
    t.write(single(saveimg));
    t.close();
end