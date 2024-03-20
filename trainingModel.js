let video;
let poseNet;
let pose, skeleton;
let brain;
let state = 'inactive';
let targetLabel;
const frameWidth = 800;
const frameHeight = 600;

const waitingTime = 5000;
const collectingTime = 10000;

function keyPressed() {
  if (keyCode === ENTER) {
    brain.saveData();
  }

  // Space key pressed
  if (keyCode == 32) {
    // normalizing the data to be between 0 and 1 (helps the model to learn more effectively and accurately)
    brain.normalizeData();
    // epochs - how many times the model will go through the each piece of the training data
    brain.train({ epochs: 50 }, finishedTraining);
  }

  // if someone pressed one of the poses keys, start the training process for that pose
  const isPoseKey = targetPoses.some(targetPose => targetPose.key === key.toUpperCase());
  if (isPoseKey) {
    // firstly wait for 5 seconds(preparation time to get ready for the pose) 
    // and set the target pose name used for training the model later on 
    state = 'waiting';
    targetLabel = targetPoses.find(targetPose => targetPose.key === key.toUpperCase()).poseName;

    // after 5 seconds, start collecting the data for the target pose and collect it for 10 seconds
    setTimeout(() => {
      state = 'collecting';
    }, waitingTime);

    // after 15 seconds, set the state to inactive(data collection process is finished for the target pose)
    setTimeout(() => {
      state = 'inactive';
    }, waitingTime + collectingTime);
  }
}

// target poses and their keys(keys used to trigger collecting the data of certain pose for the model)
const targetPoses = [
  { key: 'D', poseName: 'Hands & legs separated' },
  { key: 'U', poseName: 'Half squat - upper body flex' },
  { key: 'B', poseName: 'Stretch Hands straight up' },
  { key: 'R', poseName: 'Candle Pose' },
  { key: 'C', poseName: 'Huddle Pose' },
  { key: 'T', poseName: 'Left-leg & right-arm triangles' },
  { key: 'H', poseName: 'Halo around your head' },
  { key: 'S', poseName: 'Stretch back (right side)' },
  { key: 'F', poseName: 'Stretch back (left side)' }
];

function setup() {
  createCanvas(frameWidth, frameHeight);

  video = createCapture(VIDEO);
  video.size(frameWidth, frameHeight);
  video.hide();

  poseNet = ml5.poseNet(video, () => console.log('PoseNet loaded!'));
  poseNet.on('pose', getPoses);

  const options = {
    inputs: 34,
    outputs: 9,
    task: 'classification',
    debug: true
  };

  brain = ml5.neuralNetwork(options);

  // loading data from the file that contains the data used for model training
  brain.loadData('data.json', dataReady);
}

function dataReady() {
  console.log('Data for training the model is ready!');
}

function finishedTraining() {
  console.log('Model is ready to classify!');
  // once the model is trained, saving the model
  brain.save();
}

function draw() {
  applyMoreNaturalVideoFeel();
  image(video, 0, 0);
  noStroke();

  // drawing skeleton
  if (pose) {
    // drawing lines between the keypoints
    for (let i = 0; i < skeleton.length; i++) {
      const a = skeleton[i][0];
      const b = skeleton[i][1];
      strokeWeight(2);
      stroke(0);
      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    // drawing keypoints as little circles
    for (let i = 0; i < pose.keypoints.length; i++) {
      const x = pose.keypoints[i].position.x;
      const y = pose.keypoints[i].position.y;
      fill(0);
      stroke(255);
      ellipse(x, y, 16, 16);
    }
  }

  stroke(255);
  // ellipse in the bottom right corner of the screen
  // gray - inactive, yellow - waiting, red - collecting state
  if (state == 'inactive') {
    fill('#dddddd');
  } else if (state == 'waiting') {
    fill('#f5ed11');
  } else if (state == 'collecting') {
    fill('#f51111');
  }

  ellipse(60, 540, 100);
}

function getPoses(poses) {
  if (poses.length > 0) {
    // getting the first pose poseNet detects
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;

    // if the state collecting, adding the data to the model for training
    if (state === 'collecting') {
      addDataToTheModel();
    }
  }
}

function addDataToTheModel() {
  // formatting inputs for the model to be trained (keypoints of the pose)
  let inputs = [];
  let target = [];
  if (pose) {
    for (let i = 0; i < pose.keypoints.length; i++) {
      const x = pose.keypoints[i].position.x;
      const y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    // formatting the target label for the model to be trained (the pose name)
    target.push(targetLabel);
  }

  brain.addData(inputs, target);
}

// flipping the screen according to the y axis (more natural feel)
function applyMoreNaturalVideoFeel() {
  // starting to paint the image from the right side of the screen
  translate(video.width, 0);
  // setting the direction of painting the image to the screen to right-to-left
  scale(-1, 1);
}
