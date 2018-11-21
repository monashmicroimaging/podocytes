/*
 * Fiji/ImageJ macro script to set channel LUTS for validation output.
 * 
 * See acknowledging Monash Micro Imaging in your research:
 * https://platforms.monash.edu/mmi/index.php?option=com_content&view=article&id=124&Itemid=244
 * 
 * To run this macro:
 * 1. Open Fiji's script editor (File > New > Script; or use the keyboard shortcut "[")
 * 2. Open the macro file (can also drag and drop .ijm files to open in Fiji)
 * 3. Click the "Run" button from the script editor, or use the keyboard shortcut Control+R
 */

setBatchMode(true);
print("Reset lookup tables for each channel in podocyte validation images.");
inputDirectory = getDirectory("Choose a Directory");
processFolder(inputDirectory);
print("Finished.");

// function to set LUTs for each channel in validation output
function saveChannelLUTS(nChannels) {
	LUTS = newArray("glasbey inverted", "Grays", "Red", "Green", "Grays");
	for (i = 0; i < nChannels; i++) {
		channelIndex = i + 1;
		Stack.setChannel(channelIndex);
		run(LUTS[i]);
	}
	//run("Channels Tool...");
	Stack.setDisplayMode("composite");
	Stack.setActiveChannels("11100");
	run("Save");
}

// function to process each image series in the file
function processFile(filename) {
	open(filename);
	getDimensions(width, height, channels, slices, frames);
	if (channels == 5) {
		saveChannelLUTS(channels);
		print(filename);
	}
	close();
}

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	print("Processing files in directory: " + input);
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], ".tif"))
			filename = input + File.separator + list[i];
			processFile(filename);
	}
}