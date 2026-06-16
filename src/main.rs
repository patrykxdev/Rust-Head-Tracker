use opencv::{Result, core, highgui, imgproc, objdetect, prelude::*, videoio};

fn detect_face(
    frame: &Mat,
    detector: &mut objdetect::CascadeClassifier,
) -> Result<Option<core::Rect>> {
    //return rect if face is found otherwise return none

    let mut grey = Mat::default();
    imgproc::cvt_color_def(frame, &mut grey, imgproc::COLOR_BGR2GRAY)?;
    //convert to black and white for faster processing

    let mut faces = core::Vector::<core::Rect>::new();
    detector.detect_multi_scale(
        &grey,
        &mut faces,
        1.1,                       // scale factor (how much the image is resized)
        5,                         // min neighbors (how many detections make a face real)
        0,                         // flags (unnecessary to change)
        core::Size::new(100, 100), // min face size
        core::Size::new(0, 0),     // max face size, (no limit)
    )?;

    if faces.len() > 0 {
        Ok(Some(faces.get(0)?))
        //return first face found
    } else {
        Ok(None)
    }
}

fn main() -> Result<()> {
    //opencv has Result to handle errors, can use instead of anyhow

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    //0 defaults to iphone cameraa, but removed phone from camera list so 0 is now mac camera

    let mut face_detector =
        objdetect::CascadeClassifier::new("haarcascade_frontalface_default.xml")?;

    highgui::named_window("tracker", highgui::WINDOW_AUTOSIZE)?;
    //create window

    let mut frame = Mat::default();
    //store video frames as matrices

    loop {
        cam.read(&mut frame)?;
        //read current frame

        if let Some(face_rect) = detect_face(&frame, &mut face_detector)? {
            //if there is a face create the face_rect var and run lines below else ignore

            let x = face_rect.x + (face_rect.width / 2);
            let y = face_rect.y + (face_rect.height / 5);
            //forehead position

            let forehead = core::Point::new(x, y);
            //create forehead point

            imgproc::circle(
                &mut frame,
                forehead,
                10,                                      //radius
                core::Scalar::new(255.0, 0.0, 0.0, 0.0), //blue
                -1,                                      //filled circle
                imgproc::LINE_8, //tiles are connected if touch on corners and sides
                0,
            )?;
        }
        highgui::imshow("tracker", &frame)?;
        //(doesn't display yet, line below shows the images on the screen)

        let key = highgui::wait_key(10)?;
        //displays new image every 10ms

        if key == 27 {
            //esc key
            break;
        }
    }
    Ok(())
}
