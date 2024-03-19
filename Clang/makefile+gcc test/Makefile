CC=gcc
CFLAGS=-g -Wall
OBJS=main.o foo.o bar.o
TARGET=APP.OUT

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS)

main.o: foo.h bar.h main.c
foo.o: foo.h foo.c
bar.o: bar.h bar.c

clean:
	rm -f *.o foo
	rm -f $(TARGET)

