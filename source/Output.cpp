#include "Output.h"

#ifdef _WIN32
    #include <windows.h>
#elif __linux__
    #include <X11/Xlib.h>
#elif __APPLE__
    #include <ApplicationServices/ApplicationServices.h>
#endif

cv::Point Output::calculateCenteredCoords(int windowWidth, int windowHeight, bool singleScreen) {
    int screenWidth = 0;
    int screenHeight = 0;

#ifdef _WIN32
    // Get screen resolution on Windows
    screenWidth = GetSystemMetrics(SM_CXSCREEN);
    screenHeight = GetSystemMetrics(SM_CYSCREEN);
#elif __linux__
    // Get screen resolution on Linux using Xlib
    Display* display = XOpenDisplay(NULL);
    if (display != NULL) {
        Screen* screen = DefaultScreenOfDisplay(display);
        screenWidth = screen->width;
        screenHeight = screen->height;
        XCloseDisplay(display);
    }
#elif __APPLE__
    // Get screen resolution on macOS using CoreGraphics
    CGDirectDisplayID displayID = CGMainDisplayID();
    screenWidth = CGDisplayPixelsWide(displayID);
    screenHeight = CGDisplayPixelsHigh(displayID);
#endif

    // Check for multiple screens
    if (!singleScreen)
        screenWidth = screenWidth / 2;

    // Calculate centered coordinates
    int x = abs((screenWidth - windowWidth) / 2);
    int y = abs((screenHeight - windowHeight) / 2);

    return cv::Point(x, y);
}
